"""Run Qwen 3 locally through Ollama and write results in the shared schema.

Standard library only. Safe to stop and restart: finished calls are skipped.

Examples (from the repo root):
  python qwen/run_qwen.py --smoke
  python qwen/run_qwen.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv --dry-run
  python qwen/run_qwen.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv

Order of work: for each method, first the greedy call (temperature 0) for every pair in both
orders, then the repeated samples. So if a run is cut short, the greedy results are complete first.

Prompts, model digest and sampling settings come from the frozen root config.yaml.
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from experiments import protocol


# ------------------------------------------------------------------ config and pairs
def load_config(path: Path) -> dict:
    return protocol.load_config(path)


def load_pairs(path: Path, shards_path: Path | None, shard_label: str | None) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    owner = {}
    if shards_path:
        with open(shards_path, encoding="utf-8-sig", newline="") as fh:
            owner = {r["pair_id"]: r["shard"] for r in csv.DictReader(fh)}
    pairs = []
    for r in rows:
        if "test_id" in r:  # Upworthy
            pid, h_a, h_b = r["test_id"], r["headline_A"], r["headline_B"]
            context, dataset = "", "upworthy"  # split v2: excerpts are not model inputs
        else:  # Reddit
            pid, h_a, h_b = r["pair_id"], r["title_a"], r["title_b"]
            context, dataset = f"r/{r['subreddit']}", "reddit"
        shard = owner.get(pid, shard_label)
        if shard is None:
            raise SystemExit(f"No shard for pair {pid}: give --shard or a --shards file that lists it")
        pairs.append({"pair_id": pid, "A": h_a, "B": h_b, "context": context, "dataset": dataset, "shard": shard})
    return pairs


def build_messages(cfg: dict, method: str, pair: dict, order: str) -> list[dict]:
    return protocol.build_messages(cfg, method, pair, order)


# ------------------------------------------------------------------ parsing
def parse(text: str) -> tuple[int | None, float | None]:
    """Return (position 1/2, confidence 0.5-1.0) from the model's final answer."""
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


# ------------------------------------------------------------------ ollama
def ollama_chat(host: str, model: str, messages: list, temperature: float, seed: int,
                think: bool, num_predict: int, num_ctx: int, timeout: int) -> tuple[str, str]:
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "format": {"type": "object", "properties": {"choice": {"type": "integer", "enum": [1, 2]},
                       "confidence": {"type": "number", "minimum": 50, "maximum": 100}},
                       "required": ["choice", "confidence"], "additionalProperties": False},
            "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx}}
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    msg = out.get("message", {})
    return msg.get("content", ""), msg.get("thinking", "") or ""


def call_with_retry(fn, wait_total: int = 900):
    """Retry on connection problems (Ollama restarting, PC waking) for up to wait_total seconds."""
    delay, waited = 5, 0
    while True:
        try:
            return fn()
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            if waited >= wait_total:
                raise
            print(f"  Ollama not responding ({e}). Is the Ollama app running? Retrying in {delay}s", flush=True)
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, 60)


# ------------------------------------------------------------------ run
def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def done_keys(path: Path, expected: dict | None = None) -> set:
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid result JSON at {path}:{index + 1}; repair the incomplete line before resuming") from error
                if expected and any(r.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Existing results in {path} belong to a different protocol, model, or input file")
                keys.add((r["pair_id"], r["order"], int(r["sample_idx"])))
    return keys


def run(a) -> int:
    cfg = load_config(Path(a.config))
    protocol.validate_overrides(a, cfg)
    if any(method not in cfg["methods"] for method in a.methods):
        raise ValueError("Unknown method")
    host, model = a.host.rstrip("/"), cfg["model"]
    seed = cfg["sampling"]["seed"]
    if a.smoke:
        pair = {"pair_id": "smoke", "A": "You Won't Believe What This Dog Did Next",
                "B": "A Dog Learned to Open the Fridge", "context": "", "dataset": "upworthy", "shard": "smoke"}
        protocol.verify_model(host, cfg)
        t0 = time.time()
        text, _ = call_with_retry(lambda: ollama_chat(host, model, build_messages(cfg, "zero_shot", pair, "AB"),
                                                      0.0, seed, False, 64, 4096, a.timeout), wait_total=15)
        print(f"model={model} reply={text!r} parsed={parse(text)} seconds={time.time() - t0:.1f}")
        return 0 if parse(text)[0] else 1

    pairs = load_pairs(Path(a.pairs), Path(a.shards) if a.shards else None, a.shard)
    before = len(pairs)
    pairs = [p for p in pairs if p["dataset"] != "upworthy" or p["pair_id"] not in cfg["_dev_excluded_ids"]]
    if before != len(pairs):
        print(f"Excluded {before - len(pairs)} demonstration-related or near-test dev pairs from scoring")
    if a.limit:
        if a.limit < 0:
            raise ValueError("--limit must be nonnegative")
        pairs = pairs[: a.limit]
    if not pairs:
        raise ValueError("No eligible pairs to run")
    input_sha256 = protocol.digest(Path(a.pairs).read_bytes())
    if a.dry_run:
        calls = {method: len(pairs) * len(cfg["sampling"]["orders"]) * (cfg["methods"][method]["samples"] + 1)
                 for method in a.methods}
        for method in a.methods:
            for pair in pairs:
                for order in cfg["sampling"]["orders"]:
                    build_messages(cfg, method, pair, order)
        print(json.dumps({"pairs": len(pairs), "model": model, "model_digest": cfg["model_digest"],
                          "protocol_sha256": cfg["_protocol_sha256"], "calls_by_method": calls,
                          "sampling": cfg["sampling"], "model_calls_made": 0}, indent=2))
        return 0
    runtime = protocol.verify_model(host, cfg)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in a.methods:
        mcfg = cfg["methods"][method]
        n_samples = mcfg["samples"]
        shards = sorted({p["shard"] for p in pairs})
        files = {s: out_dir / f"results_{safe(s)}_{safe(model)}_{method}.jsonl" for s in shards}
        expected = {"protocol_sha256": cfg["_protocol_sha256"], "model_digest": cfg["model_digest"],
                    "model": model, "method": method, "input_sha256": input_sha256}
        done = {s: done_keys(f, expected) for s, f in files.items()}
        locks = {s: threading.Lock() for s in shards}
        handles = {s: open(f, "a", encoding="utf-8") for s, f in files.items()}
        jobs = []
        for idx in range(n_samples + 1):  # greedy pass first, then samples
            for p in pairs:
                for order in cfg["sampling"]["orders"]:
                    if (p["pair_id"], order, idx) not in done[p["shard"]]:
                        jobs.append((p, order, idx))
        total = len(pairs) * 2 * (n_samples + 1)
        print(f"[{method}] {total - len(jobs)}/{total} calls already done, {len(jobs)} to go", flush=True)

        def one(job):
            p, order, idx = job
            temp = cfg["sampling"]["greedy_temperature"] if idx == 0 else cfg["sampling"]["sample_temperature"]
            messages = build_messages(cfg, method, p, order)
            t0 = time.time()
            text, thinking = call_with_retry(lambda: ollama_chat(
                host, model, messages, temp, seed + idx,
                bool(mcfg.get("think", False)), int(mcfg.get("num_predict", 256)),
                int(mcfg.get("num_ctx", 4096)), a.timeout))
            pos, conf = parse(text)
            choice = None if pos is None else (order[0] if pos == 1 else order[1])
            row = {"pair_id": p["pair_id"], "shard": p["shard"], "model": model, "method": method,
                   "order": order, "sample_idx": idx, "temperature": temp, "choice": choice,
                   "stated_conf": conf, "raw_output": text, "latency": round(time.time() - t0, 3),
                   "timestamp": datetime.now(timezone.utc).isoformat(), "runner": a.runner,
                   "dataset": p["dataset"], "seed": seed + idx,
                   "protocol_sha256": cfg["_protocol_sha256"], "input_sha256": input_sha256,
                   "prompt_sha256": protocol.digest(protocol.json_bytes(messages)),
                   "confidence_valid": conf is not None, **runtime}
            if a.keep_thinking and thinking:
                row["thinking"] = thinking
            with locks[p["shard"]]:
                fh = handles[p["shard"]]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return choice is None or conf is None

        t_start, finished, fails = time.time(), 0, 0
        ex = ThreadPoolExecutor(max_workers=a.workers)
        try:
            futs = [ex.submit(one, j) for j in jobs]
            for f in as_completed(futs):
                fails += f.result()
                finished += 1
                if finished % a.progress_every == 0 or finished == len(jobs):
                    rate = finished / max(time.time() - t_start, 1e-9)
                    eta = (len(jobs) - finished) / rate / 3600 if rate else float("inf")
                    print(f"[{method}] {finished}/{len(jobs)}  {rate:.2f} calls/s  "
                          f"parse fails {fails}  ETA {eta:.1f} h", flush=True)
        except KeyboardInterrupt:
            ex.shutdown(wait=True, cancel_futures=True)  # finish calls in flight, drop the rest
            print("Stopped. Run the same command again to continue where it left off.")
            return 130
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
            for fh in handles.values():
                fh.close()
    print("All done.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs")
    ap.add_argument("--shard", help="shard label for every pair (e.g. dev); or use --shards")
    ap.add_argument("--shards", help="shard file from tools/shards.py; each pair gets its owner shard")
    ap.add_argument("--config", default=str(HERE.parent / "config.yaml"))
    ap.add_argument("--model", help="Ollama model tag; default from config")
    ap.add_argument("--methods", nargs="+", default=["zero_shot", "few_shot", "reasoning"])
    ap.add_argument("--samples", type=int, help="optional assertion of the frozen repeated-sample count")
    ap.add_argument("--sample-temperature", type=float, help="optional assertion of frozen sampling temperature")
    ap.add_argument("--seed", type=int, help="optional assertion of frozen seed")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default="results/qwen")
    ap.add_argument("--runner", default="gaeun")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (for a quick test)")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--keep-thinking", action="store_true", help="also save the reasoning text")
    ap.add_argument("--smoke", action="store_true", help="one test call, then exit")
    ap.add_argument("--dry-run", action="store_true", help="verify and render prompts without contacting a model")
    a = ap.parse_args(argv)
    if not a.smoke and not a.pairs:
        ap.error("--pairs is required unless --smoke")
    try:
        result = run(a)
    except (ValueError, FileNotFoundError, urllib.error.URLError) as error:
        ap.error(str(error))
    sys.exit(result)


if __name__ == "__main__":
    main()
