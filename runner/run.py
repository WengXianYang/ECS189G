"""Shared API runner — cloud provider equivalent of qwen/run_qwen.py.

Calls OpenAI, Anthropic, or Google models on headline pairs using the frozen
shared protocol (experiments/protocol.py + root config.yaml), writing results
in the shared schema consumed by tools/merge_check.py and tools/metrics.py.

Resume-safe: finished (pair_id, order, sample_idx) triples are skipped.
Order of work: greedy pass (sample_idx=0, temp=0) first for all pairs + orders,
then repeated samples — greedy results are always complete even if interrupted.

Examples (from repo root):
  # smoke test — one call then exit
  python runner/run.py --pairs data/prompt_dev.csv \\
      --shards data/shards/prompt_dev_shards.csv \\
      --provider openai --model gpt-4o-2024-11-20 \\
      --methods zero_shot --smoke

  # Yurun's dev pilot (D1 shard, Task 13)
  python runner/run.py \\
      --pairs data/prompt_dev.csv \\
      --shards data/shards/prompt_dev_shards.csv \\
      --provider openai --model gpt-4o-2024-11-20 \\
      --methods zero_shot few_shot reasoning \\
      --runner yurun --out-dir results/dev

  # Reddit dev predictions
  python runner/run.py \\
      --pairs reddit/reddit_pairs.csv --shard D1 \\
      --provider openai --model gpt-4o-2024-11-20 \\
      --methods zero_shot few_shot \\
      --runner yurun --out-dir results/dev

Output files: {out_dir}/results_{shard}_{model}_{method}.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from experiments import protocol  # noqa: E402  (import after path setup)


# ---------------------------------------------------------------------------
# Pair loading  (identical logic to qwen/run_qwen.py)
# ---------------------------------------------------------------------------

def load_pairs(
    path: Path,
    shards_path: Path | None,
    shard_label: str | None,
) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    owner: dict[str, str] = {}
    if shards_path:
        with open(shards_path, encoding="utf-8-sig", newline="") as fh:
            owner = {r["pair_id"]: r["shard"] for r in csv.DictReader(fh)}
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
        })
    return pairs


# ---------------------------------------------------------------------------
# Response parsing  (identical to qwen/run_qwen.py)
# ---------------------------------------------------------------------------

def parse(text: str) -> tuple[int | None, float | None]:
    """Return (position 1|2, confidence 0.5–1.0) from the model's raw output."""
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
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v > 1.0:
        v /= 100.0
    return float(max(0.5, min(1.0, v))) if v >= 0 else None


# ---------------------------------------------------------------------------
# API providers
# ---------------------------------------------------------------------------

def call_openai(
    model_id: str, messages: list[dict], temperature: float,
    seed: int, max_tokens: int, api_key: str, timeout: int,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model_id, messages=messages,
        temperature=temperature, seed=seed, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(
    model_id: str, messages: list[dict], temperature: float,
    seed: int, max_tokens: int, api_key: str, timeout: int,
) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    chat = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model_id, system=system, messages=chat,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.content[0].text if resp.content else ""


def call_google(
    model_id: str, messages: list[dict], temperature: float,
    seed: int, max_tokens: int, api_key: str, timeout: int,
) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    system_txt = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    chat = [m for m in messages if m["role"] != "system"]
    model = genai.GenerativeModel(model_id, system_instruction=system_txt)
    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in chat]
    resp = model.generate_content(
        contents,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        request_options={"timeout": timeout},
    )
    return resp.text or ""


PROVIDERS = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
}

# Default model IDs per provider (overridden by --model)
PROVIDER_DEFAULTS = {
    "openai":    "gpt-4o-2024-11-20",
    "anthropic": "claude-sonnet-4-20250514",
    "google":    "gemini-2.0-flash",
}

# API key env var per provider
PROVIDER_KEY_ENV = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
}


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------

def done_keys(path: Path) -> set[tuple[str, str, int]]:
    """Return set of (pair_id, order, sample_idx) already in path."""
    keys: set[tuple[str, str, int]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add((str(r["pair_id"]), str(r["order"]), int(r["sample_idx"])))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return keys


# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

def call_with_retry(fn, max_retries: int = 3, base_delay: float = 5.0):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [retry {attempt+1}/{max_retries}] {exc!r} — sleeping {delay:.0f}s", flush=True)
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

    pairs = load_pairs(
        Path(args.pairs),
        Path(args.shards) if args.shards else None,
        args.shard,
    )
    if args.limit:
        pairs = pairs[: args.limit]

    # Provider / model setup
    provider_name = args.provider
    if provider_name not in PROVIDERS:
        print(f"ERROR: unknown provider {provider_name!r}. Choose: {list(PROVIDERS)}", file=sys.stderr)
        return 1
    api_fn  = PROVIDERS[provider_name]
    api_key = os.environ.get(PROVIDER_KEY_ENV[provider_name], "")
    if not api_key:
        print(f"ERROR: env var {PROVIDER_KEY_ENV[provider_name]!r} is not set.", file=sys.stderr)
        return 1
    model_id = args.model or PROVIDER_DEFAULTS[provider_name]
    # Use model_id as the "model" label in output rows (what tools/ code indexes on)
    model_label = model_id

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        if method not in cfg["methods"]:
            print(f"ERROR: method {method!r} not in frozen config. Choose: {list(cfg['methods'])}", file=sys.stderr)
            return 1

        mcfg       = cfg["methods"][method]
        n_samples  = mcfg["samples"]
        max_tokens = mcfg.get("num_predict", 256)

        shards_present = sorted({p["shard"] for p in pairs})
        files  = {
            s: out_dir / f"results_{_safe(s)}_{_safe(model_label)}_{method}.jsonl"
            for s in shards_present
        }
        done   = {s: done_keys(f) for s, f in files.items()}
        locks  = {s: threading.Lock() for s in shards_present}
        handles = {s: open(f, "a", encoding="utf-8") for s, f in files.items()}

        # Greedy pass (idx=0) first, then repeated samples — so greedy is always complete
        jobs: list[tuple[dict, str, int]] = []
        for idx in range(n_samples + 1):
            for p in pairs:
                for order in cfg["sampling"]["orders"]:
                    if (p["pair_id"], order, idx) not in done[p["shard"]]:
                        jobs.append((p, order, idx))

        total   = len(pairs) * len(cfg["sampling"]["orders"]) * (n_samples + 1)
        skipped = total - len(jobs)
        print(
            f"\n[{model_label} / {method}]  "
            f"{skipped}/{total} already done, {len(jobs)} remaining",
            flush=True,
        )

        if args.smoke and jobs:
            jobs = jobs[:1]

        def process_one(job: tuple[dict, str, int]) -> bool:
            p, order, idx = job
            temp = cfg["sampling"]["greedy_temperature"] if idx == 0 \
                   else cfg["sampling"]["sample_temperature"]
            msgs = protocol.build_messages(cfg, method, p, order)
            seed = cfg["sampling"]["seed"] + idx

            t0  = time.time()
            raw = call_with_retry(
                lambda: api_fn(model_id, msgs, temp, seed, max_tokens, api_key, 60),
                max_retries=3,
                base_delay=5.0,
            )
            latency = round(time.time() - t0, 3)

            pos, conf = parse(raw)
            # Map position (1=shown first, 2=shown second) → original label A or B
            choice: str | None = None
            if pos is not None:
                choice = order[0] if pos == 1 else order[1]

            row = {
                "pair_id":     p["pair_id"],
                "shard":       p["shard"],
                "model":       model_label,
                "method":      method,
                "order":       order,
                "sample_idx":  idx,
                "temperature": temp,
                "choice":      choice,
                "stated_conf": conf,
                "raw_output":  raw,
                "latency":     latency,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "runner":      args.runner,
                "dataset":     p["dataset"],
            }

            s = p["shard"]
            with locks[s]:
                fh = handles[s]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return choice is None   # True → parse failed

        t_start  = time.time()
        finished = 0
        fails    = 0
        workers  = args.workers

        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_one, j) for j in jobs]
                for fut in as_completed(futs):
                    try:
                        fails += fut.result()
                    except Exception as exc:
                        fails += 1
                        print(f"  [error] {exc!r}", flush=True)
                    finished += 1
                    if finished % 50 == 0 or finished == len(jobs):
                        elapsed = max(time.time() - t_start, 1e-9)
                        rate    = finished / elapsed
                        eta_m   = (len(jobs) - finished) / rate / 60 if rate else float("inf")
                        print(
                            f"  {finished}/{len(jobs)}  {rate:.1f} calls/s  "
                            f"parse_fails={fails}  ETA {eta_m:.1f} min",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\nInterrupted — re-run the same command to resume.", flush=True)
            for fh in handles.values():
                fh.close()
            return 130

        for fh in handles.values():
            fh.close()

        if args.smoke:
            for f in files.values():
                if f.exists():
                    with open(f) as fh:
                        for line in fh:
                            r = json.loads(line)
                            print("\n--- SMOKE TEST RESULT ---")
                            print(f"pair_id:     {r['pair_id']}")
                            print(f"model:       {r['model']}")
                            print(f"method:      {r['method']}")
                            print(f"order:       {r['order']}")
                            print(f"choice:      {r['choice']}")
                            print(f"stated_conf: {r['stated_conf']}")
                            print(f"latency:     {r['latency']}s")
                            print(f"raw_output:  {r['raw_output'][:300]}")
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
        "--provider", required=True, choices=list(PROVIDERS),
        help="API provider: openai | anthropic | google",
    )
    ap.add_argument(
        "--model", default=None,
        help="Exact model ID (default: provider's current default). "
             "Must be a currently available model.",
    )
    ap.add_argument(
        "--methods", nargs="+", default=["zero_shot", "few_shot", "reasoning"],
        help="Methods to run. Must match keys in frozen config.yaml (default: all three).",
    )
    ap.add_argument(
        "--out-dir", default="results/api",
        help="Output directory for JSONL result files (default: results/api)",
    )
    ap.add_argument(
        "--runner", default="yurun",
        help="Runner identifier written into every result row (default: yurun)",
    )
    ap.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent API calls (default: 4). Lower if rate-limited.",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Only process the first N pairs (for quick testing; 0 = no limit)",
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
    sys.exit(run(args))


if __name__ == "__main__":
    main()
