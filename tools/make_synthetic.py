"""Write fake result files in the shared schema, for testing tools before real runs exist.

Example:
  python tools/make_synthetic.py --pairs data/splits/upworthy_dev.csv --out results_synth/dev --shard dev
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from common import load_pairs
except ImportError:
    from tools.common import load_pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="X")
    ap.add_argument("--models", nargs="*", default=["fake-good", "fake-weak"])
    ap.add_argument("--methods", nargs="*", default=["zero_shot", "few_shot"])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runner", default="synthetic")
    ap.add_argument("--parse-fail-rate", type=float, default=0.01)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    truth = load_pairs(a.pairs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skill = {"fake-good": 1.2, "fake-weak": 0.2}
    for model in a.models:
        for method in a.methods:
            s = skill.get(model, 0.5) + (0.2 if method == "few_shot" else 0.0)
            path = out / f"results_{a.shard}_{model}_{method}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for _, r in truth.iterrows():
                    y = 1 if r.winner == "A" else -1
                    signal = s * y * min(1.0, 0.3 + 3 * r.gap / max(truth.gap.median(), 1e-9) / 10)
                    for order in ("AB", "BA"):
                        bias = 0.25 if order == "AB" else -0.25  # mild first-position bias
                        for idx in range(a.samples + 1):
                            temp = 0.0 if idx == 0 else 0.7
                            noise = 0.0 if idx == 0 else rng.normal(0, 1.0)
                            logit = signal + bias + noise + rng.normal(0, 0.6)
                            pA = 1 / (1 + np.exp(-logit))
                            choice = "A" if pA >= 0.5 else "B"
                            conf = round(float(np.clip(0.5 + abs(pA - 0.5) * 1.3, 0.5, 1.0)), 2)
                            if rng.random() < a.parse_fail_rate:
                                choice, conf = None, None
                            fh.write(json.dumps({
                                "pair_id": r.pair_id, "shard": a.shard, "model": model, "method": method,
                                "order": order, "sample_idx": idx, "temperature": temp, "choice": choice,
                                "stated_conf": conf, "raw_output": f"{{\"choice\": \"{choice}\"}}",
                                "latency": 0.1, "timestamp": datetime.now(timezone.utc).isoformat(),
                                "runner": a.runner, "dataset": r.dataset}) + "\n")
    print(f"wrote synthetic results to {out}")


if __name__ == "__main__":
    main()
