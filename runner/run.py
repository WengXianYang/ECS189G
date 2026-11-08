"""Shared API runner for cloud LLM headline A/B test prediction.

Calls OpenAI, Anthropic, or Google models on pairwise headline data and writes
results in the shared schema expected by tools/merge_check.py and tools/metrics.py.

Resume-safe: finished (pair_id, order, sample_idx) triples are skipped on restart.
Order of work: greedy pass (sample_idx=0, temp=0) first for all pairs, then repeated
samples — so if a run is cut short, at least the greedy results are complete.

Examples (from repo root):
  # smoke test — one call then exit
  python runner/run.py --pairs xian_data_prep/outputs/upworthy_dev.csv \\
      --shard dev --models gpt-4o --methods zero_shot --smoke

  # your dev shard (D1) — Yurun's Task 13
  python runner/run.py \\
      --pairs xian_data_prep/outputs/upworthy_dev.csv \\
      --shards data/shards/upworthy_dev_shards.csv \\
      --models gpt-4o claude-sonnet \\
      --methods zero_shot few_shot reasoning \\
      --runner yurun --out-dir results/dev

  # Reddit (no shard file needed — one shard per runner)
  python runner/run.py \\
      --pairs xian_data_prep/inputs/reddit_pairs.csv \\
      --shard D1 --models gpt-4o --methods zero_shot few_shot \\
      --runner yurun --out-dir results/dev

Output files: results/{out_dir}/results_{shard}_{model}_{method}.jsonl
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

import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Pair loading
# ---------------------------------------------------------------------------

def load_pairs(
    pairs_path: Path,
    shards_path: Path | None,
    shard_label: str | None,
) -> list[dict]:
    """Load Upworthy or Reddit pairs; assign shard from file or fixed label."""
    with open(pairs_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    owner: dict[str, str] = {}
    if shards_path:
        with open(shards_path, encoding="utf-8-sig", newline="") as fh:
            owner = {r["pair_id"]: r["shard"] for r in csv.DictReader(fh)}

    pairs = []
    for r in rows:
        if "test_id" in r:  # Upworthy
            pid  = r["test_id"]
            h_a  = r["headline_A"]
            h_b  = r["headline_B"]
            context = ""          # excerpt intentionally excluded from prompts
            dataset = "upworthy"
        else:                 # Reddit
            pid  = r["pair_id"]
            h_a  = r["title_a"]
            h_b  = r["title_b"]
            context = f"r/{r['subreddit']}"
            dataset = "reddit"

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
# Prompt building
# ---------------------------------------------------------------------------

def build_messages(cfg: dict, method: str, pair: dict, order: str) -> list[dict]:
    """Build the chat message list for one (pair, order, method).

    order='AB' → headline_1=A, headline_2=B
    order='BA' → headline_1=B, headline_2=A
    """
    first, second = (pair["A"], pair["B"]) if order == "AB" else (pair["B"], pair["A"])
    ds   = cfg["datasets"][pair["dataset"]]
    mcfg = cfg["methods"][method]

    msgs: list[dict] = [{"role": "system", "content": ds["system"]}]

    # Few-shot examples (only if method sets few_shot: true)
    if mcfg.get("few_shot"):
        for ex in cfg.get("few_shot", {}).get(pair["dataset"], []):
            user_ex = ds["user"].format(
                headline_1=ex["headline_1"],
                headline_2=ex["headline_2"],
                context=ex.get("context", ""),
            )
            msgs.append({"role": "user", "content": user_ex})
            msgs.append({
                "role": "assistant",
                "content": json.dumps({
                    "choice": ex["answer"],
                    "confidence": ex.get("confidence", 75),
                }),
            })

    user_text = ds["user"].format(
        headline_1=first,
        headline_2=second,
        context=pair["context"],
    )
    if mcfg.get("instruction"):
        user_text += "\n\n" + mcfg["instruction"]

    msgs.append({"role": "user", "content": user_text})
    return msgs


# ---------------------------------------------------------------------------
# API call — one function per provider
# ---------------------------------------------------------------------------

def call_openai(
    model_id: str,
    messages: list[dict],
    temperature: float,
    seed: int,
    max_tokens: int,
    api_key: str,
    timeout: int,
) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, timeout=timeout)
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def call_anthropic(
    model_id: str,
    messages: list[dict],
    temperature: float,
    seed: int,           # Anthropic has no seed param — accepted and ignored
    max_tokens: int,
    api_key: str,
    timeout: int,
) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    chat_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model_id,
        system=system,
        messages=chat_msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.content[0].text if resp.content else ""


def call_google(
    model_id: str,
    messages: list[dict],
    temperature: float,
    seed: int,           # Gemini has no seed — accepted and ignored
    max_tokens: int,
    api_key: str,
    timeout: int,
) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    system_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else None
    chat_msgs   = [m for m in messages if m["role"] != "system"]

    model = genai.GenerativeModel(
        model_id,
        system_instruction=system_text,
    )

    # Convert to Gemini message format
    contents = []
    for m in chat_msgs:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    resp = model.generate_content(
        contents,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
        request_options={"timeout": timeout},
    )
    return resp.text or ""


PROVIDERS = {
    "openai":    call_openai,
    "anthropic": call_anthropic,
    "google":    call_google,
}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

CHOICE_RE = re.compile(r'"?choice"?\s*[:=]\s*"?\s*([12])', re.I)
CONF_RE   = re.compile(r'"?confidence"?\s*[:=]\s*"?\s*(\d+(?:\.\d+)?)', re.I)


def parse(text: str) -> tuple[int | None, float | None]:
    """Return (position 1|2, confidence 0.5–1.0) from the model's raw output.

    position 1 = first headline shown, 2 = second headline shown.
    The caller is responsible for mapping position back to original label (A or B).
    Returns (None, None) if parsing fails.
    """
    # Strip reasoning tags (<think>…</think>) before parsing
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()

    # Try to find the last valid JSON object with {"choice": 1|2, ...}
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        c = str(obj.get("choice", "")).strip().strip('"')
        if c in ("1", "2"):
            return int(c), _parse_conf(obj.get("confidence"))

    # Fallback: regex scan
    m = CHOICE_RE.findall(text)
    if m:
        cm = CONF_RE.findall(text)
        return int(m[-1]), _parse_conf(cm[-1] if cm else None)

    return None, None


def _parse_conf(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v > 1.0:
        v /= 100.0
    return float(max(0.5, min(1.0, v))) if v >= 0 else None


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------

def done_keys(path: Path) -> set[tuple[str, str, int]]:
    """Return set of (pair_id, order, sample_idx) already written to path."""
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

def call_with_retry(fn, max_retries: int, base_delay: float):
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [retry {attempt + 1}/{max_retries}] {exc!r} — waiting {delay:.0f}s",
                  flush=True)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Core run loop
# ---------------------------------------------------------------------------

def run(args) -> int:
    cfg    = load_config(Path(args.config))
    pairs  = load_pairs(
        Path(args.pairs),
        Path(args.shards) if args.shards else None,
        args.shard,
    )
    if args.limit:
        pairs = pairs[: args.limit]

    defs      = cfg["defaults"]
    n_samples = defs["samples"]           # repeated samples besides greedy
    workers   = defs["workers"]
    max_retry = defs["max_retries"]
    base_dly  = defs["retry_base_delay"]
    prog_n    = defs.get("progress_every", 50)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if model_name not in cfg["models"]:
            print(f"ERROR: model {model_name!r} not in config. "
                  f"Available: {list(cfg['models'])}", file=sys.stderr)
            return 1

        mcfg_model = cfg["models"][model_name]
        provider   = mcfg_model["provider"]
        if provider not in PROVIDERS:
            print(f"ERROR: unknown provider {provider!r}", file=sys.stderr)
            return 1
        api_fn   = PROVIDERS[provider]
        api_key  = os.environ.get(mcfg_model["api_key_env"], "")
        if not api_key:
            print(f"ERROR: env var {mcfg_model['api_key_env']!r} is not set.", file=sys.stderr)
            return 1
        model_id = mcfg_model["model_id"]

        for method in args.methods:
            if method not in cfg["methods"]:
                print(f"ERROR: method {method!r} not in config.", file=sys.stderr)
                return 1

            mcfg_method = cfg["methods"][method]
            max_tokens  = int(mcfg_method.get("max_tokens", 256))

            # One output file per shard
            shards_present = sorted({p["shard"] for p in pairs})
            files  = {s: out_dir / f"results_{_safe(s)}_{_safe(model_name)}_{method}.jsonl"
                      for s in shards_present}
            done   = {s: done_keys(f) for s, f in files.items()}
            locks  = {s: threading.Lock() for s in shards_present}
            handles = {s: open(f, "a", encoding="utf-8") for s, f in files.items()}

            # Build job list — greedy (idx=0) first, then repeated samples
            # This guarantees greedy results are complete even if interrupted.
            jobs: list[tuple[dict, str, int]] = []
            for idx in range(n_samples + 1):
                for p in pairs:
                    for order in ("AB", "BA"):
                        key = (p["pair_id"], order, idx)
                        if key not in done[p["shard"]]:
                            jobs.append((p, order, idx))

            total = len(pairs) * 2 * (n_samples + 1)
            skipped = total - len(jobs)
            print(
                f"\n[{model_name} / {method}]  "
                f"{skipped}/{total} calls already done, {len(jobs)} remaining",
                flush=True,
            )

            if args.smoke and jobs:
                jobs = jobs[:1]   # one call only for smoke test

            def process_one(job: tuple[dict, str, int]) -> bool:
                """Execute one API call and write to JSONL. Returns True on parse fail."""
                p, order, idx = job
                temp = 0.0 if idx == 0 else defs["sample_temperature"]
                msgs = build_messages(cfg, method, p, order)

                t0 = time.time()
                raw = call_with_retry(
                    lambda: api_fn(
                        model_id, msgs, temp,
                        defs["seed"] + idx,
                        max_tokens,
                        api_key,
                        defs["timeout"],
                    ),
                    max_retries=max_retry,
                    base_delay=base_dly,
                )
                latency = round(time.time() - t0, 3)

                pos, conf = parse(raw)

                # Map position (1=first shown, 2=second shown) → original label A or B
                # order[0] is the label shown first: 'A' if order=='AB', 'B' if order=='BA'
                choice: str | None = None
                if pos is not None:
                    choice = order[0] if pos == 1 else order[1]

                row = {
                    "pair_id":     p["pair_id"],
                    "shard":       p["shard"],
                    "model":       model_name,
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

            t_start = time.time()
            finished = 0
            fails    = 0

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
                        if finished % prog_n == 0 or finished == len(jobs):
                            elapsed = max(time.time() - t_start, 1e-9)
                            rate    = finished / elapsed
                            eta_m   = (len(jobs) - finished) / rate / 60 if rate else float("inf")
                            print(
                                f"  {finished}/{len(jobs)}  "
                                f"{rate:.1f} calls/s  parse_fails={fails}  "
                                f"ETA {eta_m:.1f} min",
                                flush=True,
                            )
            except KeyboardInterrupt:
                print("\nInterrupted. Re-run the same command to resume.", flush=True)
                for fh in handles.values():
                    fh.close()
                return 130

            for fh in handles.values():
                fh.close()

            if args.smoke:
                # Print the one result for inspection
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
# Helpers
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    """Make a string safe for use in a filename."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


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
        help="Pair CSV: upworthy_dev.csv, upworthy_test.csv, or reddit_pairs.csv",
    )
    ap.add_argument(
        "--shard",
        help="Fixed shard label for every pair (e.g. 'dev'). "
             "Use when there is no shard file.",
    )
    ap.add_argument(
        "--shards",
        help="Shard CSV from tools/shards.py. Each pair gets its assigned shard label.",
    )
    ap.add_argument(
        "--config", default="runner/config.yaml",
        help="Path to config YAML (default: runner/config.yaml)",
    )
    ap.add_argument(
        "--models", nargs="+", required=True,
        help="One or more model names as defined in config (e.g. gpt-4o claude-sonnet)",
    )
    ap.add_argument(
        "--methods", nargs="+", default=["zero_shot", "few_shot", "reasoning"],
        help="Methods to run (default: zero_shot few_shot reasoning)",
    )
    ap.add_argument(
        "--out-dir", default="results/api",
        help="Output directory for JSONL result files (default: results/api)",
    )
    ap.add_argument(
        "--runner", default="yurun",
        help="Runner identifier written into each result row (default: yurun)",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Only process the first N pairs (for quick testing; 0 = no limit)",
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

