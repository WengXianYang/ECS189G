"""Shared API runner: cloud / OpenAI-compatible equivalent of qwen/run_qwen.py.

Calls OpenAI-compatible endpoints (OpenAI, or a gateway such as SoCLaaS via --base-url),
Anthropic, or Google models on headline pairs using the frozen shared protocol
(experiments/protocol.py + root config.yaml), writing results in the shared schema
consumed by tools/merge_check.py and tools/metrics.py.

Resume-safe: finished (pair_id, order, sample_idx) triples are skipped, and existing rows
recorded with different settings (model, method, effort, token budget, protocol or input)
stop the run instead of being mixed.
Order of work: greedy pass (sample_idx=0, temp=0) first for all pairs + orders,
then repeated samples, so greedy results are always complete even if interrupted.

Shard ownership (--only-shard):
  Without --only-shard every pair in --pairs is run and labelled with its owner shard.
  With --only-shard D2, only D2 pairs are run as owner copies, plus every overlap pair
  owned by another shard, labelled shard "O". merge_check uses the "O" copies only for
  the overlap-agreement check, so they never duplicate rows in merged_results.jsonl.

Examples (from repo root, PowerShell, SoCLaaS gateway):
  $env:OPENAI_BASE_URL = "https://soclaas-api.comp.nus.edu.sg/v1"

  # smoke test: one real call, printed, then exit (the row is kept and reused on resume)
  python runner/run.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv \\
      --only-shard D2 --provider openai --api-key-env SOCLAAS_API_KEY \\
      --model nemotron3.5:30b --methods zero_shot --runner xian --out-dir results/dev/xian --smoke

  # Xian's dev pilot: D2 owner rows + D1 overlap copies ("O")
  python runner/run.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv \\
      --only-shard D2 --provider openai --api-key-env SOCLAAS_API_KEY \\
      --model nemotron3.5:30b --methods zero_shot few_shot --runner xian --out-dir results/dev/xian
  python runner/run.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv \\
      --only-shard D2 --provider openai --api-key-env SOCLAAS_API_KEY \\
      --model qwen3.8:27b --methods reasoning --reasoning-effort medium \\
      --runner xian --out-dir results/dev/xian

  # Test shards A and C, reasoning greedy only
  python runner/run.py --pairs xian_data_prep/outputs/upworthy_test.csv \\
      --shards data/shards/upworthy_test_shards.csv --only-shard A C \\
      --provider openai --api-key-env SOCLAAS_API_KEY --model qwen3.8:27b \\
      --methods reasoning --reasoning-effort medium --greedy-only \\
      --runner xian --out-dir results/test/xian

Output files: {out_dir}/results_{shard}_{model}_{method}.jsonl  (shard "O" = overlap copies)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from experiments import protocol  # noqa: E402  (import after path setup)

OVERLAP_LABEL = "O"
# Claude 5.x models (and newer) reject any non-default temperature/top_p/top_k with HTTP 400.
CLAUDE_NO_TEMPERATURE = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-([5-9]|\d\d)")


# ---------------------------------------------------------------------------
# Pair loading  (same pair mapping as qwen/run_qwen.py, plus the overlap flag)
# ---------------------------------------------------------------------------

def load_pairs(
    path: Path,
    shards_path: Path | None,
    shard_label: str | None,
) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    owner: dict[str, str] = {}
    overlap: dict[str, bool] = {}
    if shards_path:
        with open(shards_path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                owner[r["pair_id"]] = r["shard"]
                overlap[r["pair_id"]] = str(r.get("overlap", "")).strip().lower() == "true"
    pairs = []
    for r in rows:
        if "test_id" in r:  # Upworthy
            pid, h_a, h_b = r["test_id"], r["headline_A"], r["headline_B"]
            context, dataset = "", "upworthy"   # split v2: excerpts are not model inputs
        else:               # Reddit
            pid, h_a, h_b = r["pair_id"], r["title_a"], r["title_b"]
            context, dataset = f"r/{r['subreddit']}", "reddit"
        shard = owner.get(pid, shard_label)
        if shard is None:
            raise SystemExit(
                f"No shard for pair {pid!r}: supply --shard or a --shards file that covers it."
            )
        pairs.append({
            "pair_id": pid,
            "A": h_a,
            "B": h_b,
            "context": context,
            "dataset": dataset,
            "shard": shard,
            "overlap": overlap.get(pid, False),
        })
    return pairs


def select_shards(pairs: list[dict], only: list[str]) -> list[dict]:
    """Keep the runner's own shards, plus other shards' overlap pairs relabelled "O"."""
    wanted = set(only)
    if OVERLAP_LABEL in wanted:
        raise SystemExit(f"--only-shard: {OVERLAP_LABEL!r} is reserved for overlap copies.")
    present = {p["shard"] for p in pairs}
    missing = wanted - present
    if missing:
        raise SystemExit(
            f"--only-shard {sorted(missing)} not in the shard file (available: {sorted(present)})."
        )
    kept = []
    for p in pairs:
        if p["shard"] in wanted:
            kept.append(p)
        elif p["overlap"]:
            kept.append({**p, "shard": OVERLAP_LABEL})
    return kept


# ---------------------------------------------------------------------------
# Response parsing  (identical to qwen/run_qwen.py)
# ---------------------------------------------------------------------------

def parse(text: str) -> tuple[int | None, float | None]:
    """Return (position 1|2, confidence 0.5-1.0) from the model's raw output."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        c = obj.get("choice")
        if type(c) is int and c in (1, 2):
            return c, _conf(obj.get("confidence"))
    return None, None


def _conf(x) -> float | None:
    """Out-of-range or invalid confidence is recorded as missing, not clipped (protocol rule)."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if 50 <= v <= 100:
        v /= 100.0
    return v if 0.5 <= v <= 1.0 else None


# ---------------------------------------------------------------------------
# API providers: each returns (final_text, meta)
# ---------------------------------------------------------------------------

_client_lock = threading.Lock()


def _openai_client(ctx: dict):
    with _client_lock:
        if ctx.get("client") is None:
            from openai import OpenAI
            # max_retries=0: retries are handled by call_with_retry so fatal errors stop at once
            ctx["client"] = OpenAI(api_key=ctx["api_key"], base_url=ctx.get("base_url") or None,
                                   timeout=ctx["timeout"], max_retries=0)
    return ctx["client"]


def call_openai(
    ctx: dict, model_id: str, messages: list[dict], temperature: float,
    seed: int, max_tokens: int,
) -> tuple[str, dict]:
    kwargs = dict(model=model_id, messages=messages, seed=seed, max_tokens=max_tokens)
    if not ctx.get("omit_temperature"):   # GPT-6.x reasoning models reject temperature
        kwargs["temperature"] = temperature
    if ctx.get("reasoning_effort"):
        kwargs["extra_body"] = {"reasoning_effort": ctx["reasoning_effort"]}
    resp = _openai_client(ctx).chat.completions.create(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    extra = getattr(msg, "model_extra", None) or {}
    thinking = (getattr(msg, "reasoning_content", None) or extra.get("reasoning_content")
                or getattr(msg, "reasoning", None) or extra.get("reasoning") or "")
    if not isinstance(thinking, str):
        thinking = json.dumps(thinking, ensure_ascii=False)
    usage = getattr(resp, "usage", None)
    return msg.content or "", {
        "finish_reason": choice.finish_reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "thinking": thinking,
    }


def call_anthropic(
    ctx: dict, model_id: str, messages: list[dict], temperature: float,
    seed: int, max_tokens: int,
) -> tuple[str, dict]:
    import anthropic
    if ctx.get("client") is None:
        ctx["client"] = anthropic.Anthropic(api_key=ctx["api_key"], timeout=ctx["timeout"],
                                            max_retries=0)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    chat = [m for m in messages if m["role"] != "system"]
    kwargs = dict(model=model_id, system=system, messages=chat, max_tokens=max_tokens)
    # Claude 5.x models return HTTP 400 for any non-default temperature (and API version
    # 2026-10-21+ for any temperature), so it is only sent when the model accepts it.
    if not ctx.get("omit_temperature"):
        kwargs["temperature"] = temperature
    extra: dict = {}
    if ctx.get("thinking"):
        extra["thinking"] = {"type": ctx["thinking"]}
    if ctx.get("reasoning_effort"):
        extra["output_config"] = {"effort": ctx["reasoning_effort"]}
    if extra:
        kwargs["extra_body"] = extra
    resp = ctx["client"].messages.create(**kwargs)
    # keep only text blocks; thinking blocks (display defaults to "omitted") are not the answer
    text = "".join(getattr(b, "text", "") or "" for b in resp.content or []
                   if getattr(b, "type", "text") == "text")
    usage = getattr(resp, "usage", None)
    return text, {
        "finish_reason": getattr(resp, "stop_reason", None),
        "prompt_tokens": getattr(usage, "input_tokens", None),
        "completion_tokens": getattr(usage, "output_tokens", None),
    }


def call_google(
    ctx: dict, model_id: str, messages: list[dict], temperature: float,
    seed: int, max_tokens: int,
) -> tuple[str, dict]:
    import google.generativeai as genai
    genai.configure(api_key=ctx["api_key"])
    system_txt = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    chat = [m for m in messages if m["role"] != "system"]
    model = genai.GenerativeModel(model_id, system_instruction=system_txt)
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in chat]
    resp = model.generate_content(
        contents,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        request_options={"timeout": ctx["timeout"]},
    )
    return resp.text or "", {}


PROVIDERS = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
}

# Default API key env var per provider (override with --api-key-env)
PROVIDER_KEY_ENV = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------

def done_keys(path: Path, expected: dict) -> set[tuple[str, str, int]]:
    """Return (pair_id, order, sample_idx) already in path; refuse rows with other settings."""
    keys: set[tuple[str, str, int]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"Invalid JSON at {path}:{n}; repair or remove that line before resuming.")
            for k, v in expected.items():
                if r.get(k) != v:
                    raise SystemExit(
                        f"{path}:{n} was recorded with {k}={r.get(k)!r}, but this run uses {v!r}. "
                        "Use matching settings or a different --out-dir; do not mix settings in one file."
                    )
            keys.add((str(r["pair_id"]), str(r["order"]), int(r["sample_idx"])))
    return keys


# ---------------------------------------------------------------------------
# Retry with exponential backoff; configuration errors stop the run
# ---------------------------------------------------------------------------

# Bad request, bad key, no permission, unknown model, invalid parameter: retrying cannot help.
NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}


class FatalAPIError(Exception):
    pass


def _status(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    return None


def call_with_retry(fn, max_retries: int = 3, base_delay: float = 5.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            status = _status(exc)
            if status in NON_RETRYABLE_STATUS:
                raise FatalAPIError(f"HTTP {status}: {exc}") from exc
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [retry {attempt+1}/{max_retries}] {exc!r}; sleeping {delay:.0f}s", flush=True)
            time.sleep(delay)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

def run(args) -> int:
    # Load the shared frozen protocol (same path as qwen/run_qwen.py uses)
    cfg_path = ROOT / "config.yaml"
    cfg = protocol.load_config(cfg_path, verify=not args.no_verify)
    print(f"Protocol: {cfg['protocol_id']} frozen {cfg['frozen_on']}", flush=True)
    print(f"Protocol SHA-256: {cfg['_protocol_sha256']}", flush=True)

    provider_name = args.provider
    if args.reasoning_effort and provider_name not in ("openai", "anthropic"):
        print("ERROR: --reasoning-effort is only supported with --provider openai "
              "(OpenAI-compatible endpoints) or anthropic (sent as output_config.effort).",
              file=sys.stderr)
        return 1
    if args.thinking and provider_name != "anthropic":
        print("ERROR: --thinking is only supported with --provider anthropic.", file=sys.stderr)
        return 1
    omit_temperature = bool(args.omit_temperature) or (
        provider_name == "anthropic" and CLAUDE_NO_TEMPERATURE.match(args.model or "") is not None)
    if omit_temperature:
        print("NOTE: temperature is not sent (the model rejects it). Rows keep the protocol "
              "temperature label and record temperature_sent=false; the API default sampling is used.",
              flush=True)
    for method in args.methods:
        if method not in cfg["methods"]:
            print(f"ERROR: method {method!r} not in frozen config. Choose: {list(cfg['methods'])}", file=sys.stderr)
            return 1

    pairs = load_pairs(
        Path(args.pairs),
        Path(args.shards) if args.shards else None,
        args.shard,
    )
    if args.only_shard:
        pairs = select_shards(pairs, args.only_shard)
        counts = Counter(p["shard"] for p in pairs)
        print("Selected pairs: " + ", ".join(f"{s}={n}" for s, n in sorted(counts.items()))
              + f"  ({OVERLAP_LABEL} = overlap copies of other shards)", flush=True)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("ERROR: no pairs to run.", file=sys.stderr)
        return 1
    input_sha256 = protocol.digest(Path(args.pairs).read_bytes())

    # Provider / model setup
    api_fn = PROVIDERS[provider_name]
    key_env = args.api_key_env or PROVIDER_KEY_ENV[provider_name]
    api_key = os.environ.get(key_env, "")
    if not api_key:
        print(f"ERROR: env var {key_env!r} is not set.", file=sys.stderr)
        return 1
    base_url = None
    if provider_name == "openai":
        base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or None
    elif args.base_url:
        print("ERROR: --base-url is only supported with --provider openai.", file=sys.stderr)
        return 1
    api_base = base_url or provider_name
    model_id = args.model
    print(f"Model: {model_id}  via {api_base}  reasoning_effort={args.reasoning_effort}", flush=True)

    ctx = {"api_key": api_key, "base_url": base_url, "timeout": args.timeout,
           "reasoning_effort": args.reasoning_effort, "client": None,
           "thinking": args.thinking, "omit_temperature": omit_temperature}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        mcfg       = cfg["methods"][method]
        n_samples  = 0 if args.greedy_only else mcfg["samples"]
        max_tokens = args.max_tokens or int(mcfg.get("num_predict", 256))
        if mcfg.get("think") and not args.reasoning_effort and args.thinking != "adaptive":
            print(f"WARNING: method {method!r} expects a reasoning model, but no --reasoning-effort "
                  "was given; the endpoint's default thinking behaviour will be used.", flush=True)

        shards_present = sorted({p["shard"] for p in pairs})
        files = {
            s: out_dir / f"results_{_safe(s)}_{_safe(model_id)}_{method}.jsonl"
            for s in shards_present
        }
        expected = {
            "model": model_id, "method": method, "reasoning_effort": args.reasoning_effort,
            "max_tokens": max_tokens, "protocol_sha256": cfg["_protocol_sha256"],
            "input_sha256": input_sha256,
        }
        if provider_name == "anthropic" or omit_temperature:
            expected.update({"thinking_mode": args.thinking, "temperature_sent": not omit_temperature})
        done = {s: done_keys(f, expected) for s, f in files.items()}

        # Greedy pass (idx=0) first, then repeated samples, so greedy is always complete
        jobs: list[tuple[dict, str, int]] = []
        for idx in range(n_samples + 1):
            for p in pairs:
                for order in cfg["sampling"]["orders"]:
                    if (p["pair_id"], order, idx) not in done[p["shard"]]:
                        jobs.append((p, order, idx))

        total   = len(pairs) * len(cfg["sampling"]["orders"]) * (n_samples + 1)
        skipped = total - len(jobs)
        print(
            f"\n[{model_id} / {method}]  samples={n_samples}  max_tokens={max_tokens}  "
            f"{skipped}/{total} already done, {len(jobs)} remaining",
            flush=True,
        )
        if args.smoke:
            jobs = jobs[:1]
        if not jobs:
            if args.smoke:
                print("Smoke: nothing left to run for this method.")
                return 0
            continue

        locks   = {s: threading.Lock() for s in shards_present}
        handles = {s: open(f, "a", encoding="utf-8") for s, f in files.items()}
        written: list[dict] = []
        stop = threading.Event()   # set after a fatal error so queued jobs make no more calls
        first_fatal: list[str] = []

        def process_one(job: tuple[dict, str, int]) -> tuple[bool, bool]:
            if stop.is_set():
                raise FatalAPIError("skipped after an earlier fatal error")
            p, order, idx = job
            temp = cfg["sampling"]["greedy_temperature"] if idx == 0 \
                   else cfg["sampling"]["sample_temperature"]
            msgs = protocol.build_messages(cfg, method, p, order)
            seed = cfg["sampling"]["seed"] + idx

            t0 = time.time()
            try:
                raw, meta = call_with_retry(
                    lambda: api_fn(ctx, model_id, msgs, temp, seed, max_tokens),
                    max_retries=3,
                    base_delay=5.0,
                )
            except FatalAPIError as exc:
                if not stop.is_set():
                    first_fatal.append(str(exc))
                stop.set()
                raise
            latency = round(time.time() - t0, 3)

            pos, conf = parse(raw)
            # Map position (1=shown first, 2=shown second) to original label A or B
            choice: str | None = None
            if pos is not None:
                choice = order[0] if pos == 1 else order[1]
            thinking = meta.get("thinking") or ""

            row = {
                "pair_id":          p["pair_id"],
                "shard":            p["shard"],
                "model":            model_id,
                "method":           method,
                "order":            order,
                "sample_idx":       idx,
                "temperature":      temp,
                "choice":           choice,
                "stated_conf":      conf,
                "raw_output":       raw,
                "latency":          latency,
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "runner":           args.runner,
                "dataset":          p["dataset"],
                "seed":             seed,
                "reasoning_effort": args.reasoning_effort,
                "thinking_mode":    args.thinking,
                "temperature_sent": not omit_temperature,
                "max_tokens":       max_tokens,
                "finish_reason":    meta.get("finish_reason"),
                "prompt_tokens":    meta.get("prompt_tokens"),
                "completion_tokens": meta.get("completion_tokens"),
                "thinking_chars":   len(thinking),
                "api_base":         api_base,
                "protocol_sha256":  cfg["_protocol_sha256"],
                "input_sha256":     input_sha256,
                "prompt_sha256":    protocol.digest(protocol.json_bytes(msgs)),
                "confidence_valid": conf is not None,
            }
            if args.keep_thinking and thinking:
                row["thinking"] = thinking

            s = p["shard"]
            with locks[s]:
                fh = handles[s]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                written.append(row)
            return (choice is None or conf is None), meta.get("finish_reason") == "length"

        t_start = time.time()
        finished = fails = errors = truncated = 0
        ex = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futs = [ex.submit(process_one, j) for j in jobs]
            for fut in as_completed(futs):
                try:
                    bad, trunc = fut.result()
                    fails += bad
                    truncated += trunc
                except FatalAPIError as exc:
                    print(f"\nERROR: {first_fatal[0] if first_fatal else exc}\nStopped: retrying cannot fix this. Check the model ID, "
                          "API key, base URL and parameters, then re-run the same command.",
                          file=sys.stderr, flush=True)
                    return 1
                except Exception as exc:
                    errors += 1
                    print(f"  [error] {exc!r}", flush=True)
                finished += 1
                if finished % 50 == 0 or finished == len(jobs):
                    elapsed = max(time.time() - t_start, 1e-9)
                    rate    = finished / elapsed
                    eta_m   = (len(jobs) - finished) / rate / 60 if rate else float("inf")
                    print(
                        f"  {finished}/{len(jobs)}  {rate:.1f} calls/s  "
                        f"parse_fails={fails}  errors={errors}  hit_max_tokens={truncated}  "
                        f"ETA {eta_m:.1f} min",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\nInterrupted; re-run the same command to resume.", flush=True)
            return 130
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
            for fh in handles.values():
                fh.close()

        print(f"[{model_id} / {method}] finished: {fails} parse/confidence failures, "
              f"{errors} calls failed after retries (re-run to fill them), "
              f"{truncated} stopped at max_tokens={max_tokens}.", flush=True)
        if truncated:
            print("  Some outputs hit the token limit; if they also failed to parse, "
                  "raise --max-tokens in a fresh --out-dir.", flush=True)

        if args.smoke:
            for r in written:
                print("\n--- SMOKE TEST RESULT ---")
                for k in ("pair_id", "shard", "model", "method", "order", "choice", "stated_conf",
                          "finish_reason", "prompt_tokens", "completion_tokens", "thinking_chars",
                          "latency"):
                    print(f"{k + ':':19}{r[k]}")
                print(f"{'raw_output:':19}{r['raw_output'][:300]}")
            return 0

    print("\nAll done.", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--pairs", required=True,
        help="Pair CSV: data/prompt_dev.csv, upworthy_test.csv, or reddit/reddit_pairs.csv",
    )
    ap.add_argument(
        "--shard",
        help="Fixed shard label for all pairs (e.g. 'D1'). Use when there is no shard file.",
    )
    ap.add_argument(
        "--shards",
        help="Shard CSV from tools/shards.py. Each pair gets its assigned owner shard.",
    )
    ap.add_argument(
        "--only-shard", nargs="+", metavar="SHARD",
        help="Run only these owner shards (e.g. D2, or A C), plus other shards' overlap "
             "pairs labelled 'O'. Requires --shards.",
    )
    ap.add_argument(
        "--provider", required=True, choices=list(PROVIDERS),
        help="API provider: openai (also any OpenAI-compatible gateway) | anthropic | google",
    )
    ap.add_argument(
        "--model", required=True,
        help="Exact model ID, e.g. nemotron3.5:30b or qwen3.8:27b. Use canonical IDs, not aliases.",
    )
    ap.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL, else api.openai.com)",
    )
    ap.add_argument(
        "--api-key-env", default=None,
        help="Env var holding the API key (default: OPENAI_API_KEY / ANTHROPIC_API_KEY / "
             "GOOGLE_API_KEY). The key itself is never written to results.",
    )
    ap.add_argument(
        "--reasoning-effort", default=None,
        help="OpenAI: sent as reasoning_effort (e.g. none | low | medium | xhigh). Anthropic: sent "
             "as output_config.effort (low | medium | high | xhigh | max). Recorded in every row.",
    )
    ap.add_argument(
        "--thinking", choices=["disabled", "adaptive"], default=None,
        help="Anthropic only: sent as thinking.type. Haiku/Sonnet 5.5 accept 'disabled' at effort "
             "high or below; Opus 5.5 and Fable 5.x always think (use 'adaptive' or omit).",
    )
    ap.add_argument(
        "--omit-temperature", action="store_true",
        help="Do not send temperature (automatic for Claude 5.x models, which reject it).",
    )
    ap.add_argument(
        "--max-tokens", type=int, default=None,
        help="Output token budget, including thinking tokens (default: num_predict from "
             "config.yaml: 64 zero/few-shot, 2048 reasoning).",
    )
    ap.add_argument(
        "--methods", nargs="+", default=["zero_shot", "few_shot", "reasoning"],
        help="Methods to run. Must match keys in frozen config.yaml (default: all three).",
    )
    ap.add_argument(
        "--greedy-only", action="store_true",
        help="Run only the greedy call (sample_idx 0) for each pair and order.",
    )
    ap.add_argument(
        "--out-dir", default="results/api",
        help="Output directory for JSONL result files; use one folder per runner "
             "(e.g. results/dev/xian).",
    )
    ap.add_argument(
        "--runner", required=True,
        help="Runner identifier written into every result row (gaeun / yurun / xian)",
    )
    ap.add_argument(
        "--workers", type=int, default=2,
        help="Concurrent API calls (default: 2). Lower if rate-limited (HTTP 429).",
    )
    ap.add_argument(
        "--timeout", type=int, default=300,
        help="Seconds per API call (default: 300; reasoning calls can be slow).",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Only process the first N selected pairs (for quick testing; 0 = no limit)",
    )
    ap.add_argument(
        "--keep-thinking", action="store_true",
        help="Also save the model's reasoning text in each row (larger files).",
    )
    ap.add_argument(
        "--no-verify", action="store_true",
        help="Skip EXPERIMENT_LOCK.json verification (use only for local testing)",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Run exactly 1 call, print the result, then exit",
    )
    args = ap.parse_args(argv)
    if not args.shard and not args.shards:
        ap.error("Supply either --shard (fixed label) or --shards (shard CSV file).")
    if args.only_shard and not args.shards:
        ap.error("--only-shard needs --shards (the shard file says which pairs are overlap pairs).")
    if args.limit < 0:
        ap.error("--limit must be nonnegative")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
