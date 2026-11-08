"""Split a pair file into equal stratified shards, plus an overlap set every runner also runs.

Upworthy strata: year x effect-size tercile x winner. Reddit strata: first_posted x score-gap tercile x winner.
Overlap pairs also belong to a normal shard; the shard owner's copy is the one used in analysis,
the other runners' copies are only for checking that pipelines agree.

Examples:
  # test phase: 3 shards + 150 overlap pairs
  python tools/shards.py --pairs data/splits/upworthy_test.csv --labels A B C --overlap 150 \
      --out data/shards/upworthy_test_shards.csv
  python tools/shards.py --pairs reddit/reddit_pairs.csv --labels A B C --overlap 0 \
      --out data/shards/reddit_shards.csv
  # dev pilot: two API halves + 60 overlap pairs
  python tools/shards.py --pairs data/splits/upworthy_dev.csv --labels D1 D2 --overlap 60 \
      --out data/shards/upworthy_dev_shards.csv
Output columns: pair_id, dataset, stratum, shard, overlap
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from common import load_pairs
except ImportError:
    from tools.common import load_pairs


def strata(t: pd.DataFrame) -> pd.Series:
    tercile = pd.qcut(t["gap"].rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    win = "_win" + t["winner"].astype(str)
    if t["dataset"].iloc[0] == "upworthy":
        return t["year"].astype(str) + "_" + tercile + win
    return "first" + t["first_posted"].astype(str) + "_" + tercile + win


def assign(t: pd.DataFrame, labels: list[str], n_overlap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = t.copy()
    t["stratum"] = strata(t)
    t = t.sample(frac=1, random_state=seed).sort_values("stratum", kind="mergesort")
    # round-robin inside each stratum, starting at a random label so no shard is always first
    shard = np.empty(len(t), dtype=object)
    pos = 0
    for _, g in t.groupby("stratum", sort=True):
        start = int(rng.integers(len(labels)))
        for i in range(len(g)):
            shard[pos + i] = labels[(start + i) % len(labels)]
        pos += len(g)
    t["shard"] = shard
    t["overlap"] = False
    if n_overlap:
        # stratified overlap sample, proportional to stratum size
        frac = n_overlap / len(t)
        picked = t.groupby("stratum", group_keys=False).apply(
            lambda g: g.sample(n=int(round(len(g) * frac)), random_state=seed))
        if len(picked) > n_overlap:
            picked = picked.sample(n=n_overlap, random_state=seed)
        elif len(picked) < n_overlap:
            rest = t.drop(picked.index)
            picked = pd.concat([picked, rest.sample(n=n_overlap - len(picked), random_state=seed)])
        t.loc[picked.index, "overlap"] = True
    return t[["pair_id", "dataset", "stratum", "shard", "overlap"]].sort_values("pair_id").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--labels", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t = load_pairs(a.pairs)
    s = assign(t, a.labels, a.overlap, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out, index=False, lineterminator="\n")
    merged = s.merge(t[["pair_id", "winner", "gap"]], on="pair_id")
    summary = {
        "source": str(a.pairs),
        "source_sha256": hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest(),
        "out_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "seed": a.seed, "n_pairs": len(s), "n_overlap": int(s["overlap"].sum()),
        "per_shard": merged.groupby("shard").agg(
            pairs=("pair_id", "size"), winner_A_share=("winner", lambda w: round((w == "A").mean(), 3)),
            median_gap=("gap", "median"), overlap=("overlap", "sum")).reset_index().to_dict("records"),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
