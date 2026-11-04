"""Run Qwen 3 locally through Ollama and write results in the shared schema.

Standard library only. Safe to stop and restart: finished calls are skipped.

Examples (from the repo root):
  python qwen/run_qwen.py --smoke
  python qwen/run_qwen.py --pairs data/splits/upworthy_dev.csv --shard dev
  python qwen/run_qwen.py --pairs data/splits/upworthy_test.csv --shards data/shards/upworthy_test_shards.csv

Order of work: for each method, first the greedy call (temperature 0) for every pair in both
orders, then the repeated samples. So if a run is cut short, the greedy results are complete first.

Prompts come from --config (JSON, or YAML if PyYAML is installed). Replace the placeholder
prompts in qwen/qwen_config.json with the frozen team prompts before real runs.
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ config and pairs
def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # needs PyYAML
        return yaml.safe_load(text)
    return json.loads(text)


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
    m = cfg["methods"][method]
    first, second = (pair["A"], pair["B"]) if order == "AB" else (pair["B"], pair["A"])
    ds = cfg["datasets"][pair["dataset"]]
    msgs = [{"role": "system", "content": ds["system"]}]
    for ex in (cfg.get("few_shot", {}).get(pair["dataset"], []) if m.get("few_shot") else []):
        msgs.append({"role": "user", "content": ds["user"].format(
            headline_1=ex["headline_1"], headline_2=ex["headline_2"], context=ex.get("context", ""))})
        msgs.append({"role": "assistant", "content": json.dumps({"choice": ex["answer"], "confidence": ex.get("confidence", 75)})})
    user = ds["user"].format(headline_1=first, headline_2=second, context=pair["context"])
    if m.get("instruction"):
        user += "\n\n" + m["instruction"]
    msgs.append({"role": "user", "content": user})
    return msgs


# ------------------------------------------------------------------ parsing
CHOICE_RE = re.compile(r'"?choice"?\s*[:=]\s*"?\s*([12])', re.I)
CONF_RE = re.compile(r'"?confidence"?\s*[:=]?\s*"?\s*(\d+(?:\.\d+)?)', re.I)


def parse(text: str) -> tuple[int | None, float | None]:
    """Return (position 1/2, confidence 0.5-1.0) from the model's final answer."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        c = str(obj.get("choice", "")).strip().strip('"')
        if c in ("1", "2"):
            return int(c), _conf(obj.get("confidence"))
    m = CHOICE_RE.findall(text)
    if m:
        cm = CONF_RE.findall(text)
        return int(m[-1]), _conf(cm[-1] if cm else None)
    return None, None


def _conf(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v > 1:
        v /= 100.0
    return max(0.5, min(1.0, v)) if v >= 0 else None


# ------------------------------------------------------------------ ollama
def ollama_chat(host: str, model: str, messages: list, temperature: float, seed: int,
                think: bool, num_predict: int, num_ctx: int, timeout: int) -> tuple[str, str]:
    body = {"model": model, "messages": messages, "stream": False, "think": think,
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


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line from a crash; it gets redone
                keys.add((r["pair_id"], r["order"], int(r["sample_idx"])))
    return keys


def run(a) -> int:
    cfg = load_config(Path(a.config))
    host, model = a.host.rstrip("/"), a.model or cfg["model"]
    if a.smoke:
        pair = {"pair_id": "smoke", "A": "You Won't Believe What This Dog Did Next",
                "B": "A Dog Learned to Open the Fridge", "context": "", "dataset": "upworthy", "shard": "smoke"}
        t0 = time.time()
        text, _ = call_with_retry(lambda: ollama_chat(host, model, build_messages(cfg, "zero_shot", pair, "AB"),
                                                      0.0, 0, False, 64, 2048, a.timeout), wait_total=15)
        print(f"model={model} reply={text!r} parsed={parse(text)} seconds={time.time() - t0:.1f}")
        return 0 if parse(text)[0] else 1

    pairs = load_pairs(Path(a.pairs), Path(a.shards) if a.shards else None, a.shard)
    if a.limit:
        pairs = pairs[: a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in a.methods:
        mcfg = cfg["methods"][method]
        n_samples = mcfg.get("samples", a.samples)
        shards = sorted({p["shard"] for p in pairs})
        files = {s: out_dir / f"results_{safe(s)}_{safe(model)}_{method}.jsonl" for s in shards}
        done = {s: done_keys(f) for s, f in files.items()}
        locks = {s: threading.Lock() for s in shards}
        handles = {s: open(f, "a", encoding="utf-8") for s, f in files.items()}
        jobs = []
        for idx in range(n_samples + 1):  # greedy pass first, then samples
            for p in pairs:
                for order in ("AB", "BA"):
                    if (p["pair_id"], order, idx) not in done[p["shard"]]:
                        jobs.append((p, order, idx))
        total = len(pairs) * 2 * (n_samples + 1)
        print(f"[{method}] {total - len(jobs)}/{total} calls already done, {len(jobs)} to go", flush=True)

        def one(job):
            p, order, idx = job
            temp = 0.0 if idx == 0 else a.sample_temperature
            t0 = time.time()
            text, thinking = call_with_retry(lambda: ollama_chat(
                host, model, build_messages(cfg, method, p, order), temp, a.seed + idx,
                bool(mcfg.get("think", False)), int(mcfg.get("num_predict", 256)),
                int(mcfg.get("num_ctx", 4096)), a.timeout))
            pos, conf = parse(text)
            choice = None if pos is None else (order[0] if pos == 1 else order[1])
            row = {"pair_id": p["pair_id"], "shard": p["shard"], "model": model, "method": method,
                   "order": order, "sample_idx": idx, "temperature": temp, "choice": choice,
                   "stated_conf": conf, "raw_output": text, "latency": round(time.time() - t0, 3),
                   "timestamp": datetime.now(timezone.utc).isoformat(), "runner": a.runner,
                   "dataset": p["dataset"]}
            if a.keep_thinking and thinking:
                row["thinking"] = thinking
            with locks[p["shard"]]:
                fh = handles[p["shard"]]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return choice is None

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
    ap.add_argument("--config", default=str(HERE / "qwen_config.json"))
    ap.add_argument("--model", help="Ollama model tag; default from config")
    ap.add_argument("--methods", nargs="+", default=["zero_shot", "few_shot", "reasoning"])
    ap.add_argument("--samples", type=int, default=5, help="repeated samples per order (besides greedy)")
    ap.add_argument("--sample-temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out-dir", default="results/qwen")
    ap.add_argument("--runner", default="gaeun")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (for a quick test)")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--keep-thinking", action="store_true", help="also save the reasoning text")
    ap.add_argument("--smoke", action="store_true", help="one test call, then exit")
    a = ap.parse_args(argv)
    if not a.smoke and not a.pairs:
        ap.error("--pairs is required unless --smoke")
    sys.exit(run(a))


if __name__ == "__main__":
    main()
