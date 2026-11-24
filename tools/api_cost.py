"""Actual and projected API cost from result files (uses the logged token counts).

Example:
  python tools/api_cost.py "results/dev/claude/*.jsonl" --project-pairs 2422

Spent = sum over rows of prompt_tokens * input price + completion_tokens * output price.
completion_tokens includes thinking tokens (they are billed as output).
Projection = mean cost per call (model x method) * calls a full run would make for
--project-pairs pairs (2 orders x (samples + 1); samples from config.yaml unless --greedy-only).
Prices: USD per million tokens, standard (non-batch) tier, as listed on the provider
pricing pages on 2026-11-23. Edit PRICES if they change.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

PRICES = {  # model: (input, output) USD per 1M tokens
    "claude-haiku-5-5": (0.10, 0.50),   # prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-2": (8.00, 40.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.2-sol": (2.00, 10.00),
    "gpt-6.2-astra": (8.00, 40.00),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--project-pairs", type=int, default=0, help="pairs in a full run (e.g. 2422 = dev+test+Reddit)")
    ap.add_argument("--greedy-only", action="store_true", help="project with no repeated samples")
    a = ap.parse_args(argv)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    samples = {m: v["samples"] for m, v in cfg["methods"].items()}

    agg = defaultdict(lambda: [0, 0, 0, 0])  # calls, prompt, completion, missing usage
    for f in sorted({x for p in a.results for x in glob.glob(p)}):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            k = (r["model"], r["method"])
            agg[k][0] += 1
            if r.get("prompt_tokens") is None or r.get("completion_tokens") is None:
                agg[k][3] += 1
                continue
            agg[k][1] += r["prompt_tokens"]
            agg[k][2] += r["completion_tokens"]

    total_spent = total_proj = 0.0
    print(f"{'model':18s} {'method':10s} {'calls':>6s} {'in/call':>8s} {'out/call':>8s} {'spent $':>9s} {'full run $':>10s}")
    for (model, method), (n, pin, pout, miss) in sorted(agg.items()):
        if model not in PRICES:
            print(f"{model:18s} {method:10s} {n:6d}  (no price listed)")
            continue
        ci, co = PRICES[model]
        spent = pin / 1e6 * ci + pout / 1e6 * co
        ok = n - miss
        per_call = spent / ok if ok else 0.0
        calls_full = a.project_pairs * 2 * (1 + (0 if a.greedy_only else samples.get(method, 0)))
        proj = per_call * calls_full
        total_spent += spent
        total_proj += proj
        print(f"{model:18s} {method:10s} {n:6d} {pin / max(ok, 1):8.0f} {pout / max(ok, 1):8.0f} "
              f"{spent:9.2f} {proj:10.2f}" + (f"  ({miss} rows without usage)" if miss else ""))
    print(f"{'total':18s} {'':10s} {'':6s} {'':8s} {'':8s} {total_spent:9.2f} {total_proj:10.2f}")
    if a.project_pairs:
        mode = "greedy only" if a.greedy_only else "with repeated samples"
        print(f"\nProjection: {a.project_pairs} pairs x 2 orders, {mode}.")


if __name__ == "__main__":
    main()
