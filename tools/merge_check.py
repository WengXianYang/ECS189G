"""Check result files from all runners before analysis, then write one merged file.

Checks (FAIL stops analysis, WARN is reported):
  1. JSON lines parse and have the required fields                        FAIL if any bad line
  2. No conflicting duplicates (same pair/model/method/order/sample_idx,
     same runner and shard, different choice)                              FAIL
  3. Parse-failure rate (choice is null) per model x method               WARN above --max-parse-fail
  4. Coverage: every pair in the shard file has both orders x all samples WARN, lists missing calls
  5. Overlap agreement: greedy choices on overlap pairs match across
     runners                                                               FAIL below --min-agree
  6. Shard accuracy: each shard's accuracy is inside the other shards'
     95% CI                                                                WARN if not

Overlap copies are rows whose pair is flagged overlap=True in the shard file and whose
`shard` field differs from the pair's assigned shard (e.g. "O"). Only the owner's rows go into
the merged file.

Example:
  python tools/merge_check.py --results "results/test/*.jsonl" \
      --shards data/shards/upworthy_test_shards.csv --pairs data/splits/upworthy_test.csv \
      --out reports/merge_test
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from common import KEY, REQUIRED, load_pairs, load_results
    from metrics import predictions_from_results, score_frame, wilson
except ImportError:
    from tools.common import KEY, REQUIRED, load_pairs, load_results
    from tools.metrics import predictions_from_results, score_frame, wilson


def run(results, shards_path, pairs_path, out, samples, max_parse_fail, min_agree, runner_from_file):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = load_results(results)
    shards = pd.read_csv(shards_path, dtype={"pair_id": str})
    checks = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    # 1. schema
    bad = res[res["_bad_json"]]
    res = res[~res["_bad_json"]].copy()
    missing_fields = res[REQUIRED[:6]].isna().any(axis=1)
    add("1 schema", "FAIL" if len(bad) or missing_fields.any() else "PASS",
        f"{len(bad)} bad JSON lines, {int(missing_fields.sum())} rows missing required fields")
    res = res[~missing_fields]
    if runner_from_file or res["runner"].isna().all():
        res["runner"] = res["runner"].fillna(res["_file"].map(lambda f: Path(f).parent.name))

    # assign owner shard and overlap flag
    res = res.merge(shards[["pair_id", "shard", "overlap"]].rename(columns={"shard": "owner_shard"}),
                    on="pair_id", how="left")
    unknown = res["owner_shard"].isna()
    if unknown.any():
        add("1b pairs in shard file", "FAIL", f"{int(unknown.sum())} rows for pairs not in the shard file")
        res = res[~unknown]
    res["is_owner_copy"] = res["shard"].astype(str) == res["owner_shard"].astype(str)

    # 2. duplicates
    dkey = KEY + ["runner", "shard"]
    dup = res[res.duplicated(dkey, keep=False)]
    conflict = dup.groupby(dkey)["choice"].nunique(dropna=False)
    n_conf = int((conflict > 1).sum())
    add("2 duplicates", "FAIL" if n_conf else ("WARN" if len(dup) else "PASS"),
        f"{len(dup)} duplicate rows, {n_conf} keys with conflicting choices (exact duplicates are dropped)")
    res = res.drop_duplicates(dkey, keep="last")

    # 3. parse failures
    pf = res.groupby(["model", "method"])["choice"].apply(lambda c: c.isna().mean()).rename("parse_fail_rate")
    worst = pf.max() if len(pf) else 0.0
    add("3 parse failures", "WARN" if worst > max_parse_fail else "PASS",
        f"worst model x method parse-failure rate {worst:.2%} (limit {max_parse_fail:.0%})")

    # 4. coverage (owner copies only)
    own = res[res["is_owner_copy"]]
    expected = []
    for (m, me) in own[["model", "method"]].drop_duplicates().itertuples(index=False):
        for pid in shards["pair_id"]:
            for o in ("AB", "BA"):
                for i in range(samples + 1):
                    expected.append((pid, m, me, o, i))
    exp = pd.DataFrame(expected, columns=KEY)
    have = own[KEY].assign(sample_idx=lambda d: d["sample_idx"].astype(int))
    miss = exp.merge(have, on=KEY, how="left", indicator=True)
    miss = miss[miss["_merge"] == "left_only"].drop(columns="_merge")
    miss = miss.merge(shards[["pair_id", "shard"]], on="pair_id")
    miss.to_csv(out / "missing_calls.csv", index=False)
    add("4 coverage", "WARN" if len(miss) else "PASS",
        f"{len(miss)} of {len(exp)} expected owner calls missing (see missing_calls.csv)")

    # 5. overlap agreement
    # each runner's overlap-run copy (shard != owner); fall back to the owner copy if that runner has none
    ov = res[(res["overlap"] == True) & res["greedy"] & res["choice"].notna()]  # noqa: E712
    ov = ov.sort_values("is_owner_copy", kind="mergesort")
    agree_rows = []
    for (m, me), g in ov.groupby(["model", "method"]):
        piv = g.pivot_table(index=["pair_id", "order"], columns="runner", values="choice", aggfunc="first")
        for r1, r2 in itertools.combinations(piv.columns, 2):
            both = piv[[r1, r2]].dropna()
            if len(both):
                agree_rows.append({"model": m, "method": me, "runner_1": r1, "runner_2": r2,
                                   "n": len(both), "agreement": float((both[r1] == both[r2]).mean())})
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(out / "overlap_agreement.csv", index=False)
    if agree.empty:
        add("5 overlap agreement", "WARN", "no overlap pairs run by 2+ runners yet")
    else:
        low = agree[agree["agreement"] < min_agree]
        add("5 overlap agreement", "FAIL" if len(low) else "PASS",
            f"min agreement {agree['agreement'].min():.1%} over {len(agree)} runner pairs (need {min_agree:.0%})")

    # 6. shard accuracy
    shard_rows = []
    if pairs_path:
        truth = load_pairs(pairs_path)
        pred = predictions_from_results(own)
        sc = score_frame(pred, truth, "combined").merge(shards[["pair_id", "shard"]], on="pair_id")
        for (m, me, s), g in sc.groupby(["model", "method", "shard"]):
            lo, hi = wilson(int(g["correct"].sum()), len(g))
            shard_rows.append({"model": m, "method": me, "shard": s, "n": len(g),
                               "accuracy": g["correct"].mean(), "ci_lo": lo, "ci_hi": hi})
    sa = pd.DataFrame(shard_rows)
    sa.to_csv(out / "shard_accuracy.csv", index=False)
    flagged = []
    for (m, me), g in sa.groupby(["model", "method"]) if len(sa) else []:
        for a, b in itertools.permutations(g.itertuples(index=False), 2):
            if not (b.ci_lo <= a.accuracy <= b.ci_hi):
                flagged.append(f"{m}/{me}: shard {a.shard} {a.accuracy:.3f} outside shard {b.shard} CI")
    add("6 shard accuracy", "WARN" if flagged else ("PASS" if len(sa) else "SKIP"),
        "; ".join(sorted(set(flagged))[:5]) if flagged else "all shards inside each other's 95% CI")

    # merged output (owner copies only)
    keep = [c for c in own.columns if not c.startswith("_") and c not in ("owner_shard", "is_owner_copy", "conf", "greedy")]
    own[keep].to_json(out / "merged_results.jsonl", orient="records", lines=True, force_ascii=False)

    ch = pd.DataFrame(checks)
    ch.to_csv(out / "checks.csv", index=False)
    with open(out / "report.md", "w", encoding="utf-8") as fh:
        fh.write("# Merge and sanity check\n\n")
        fh.write(ch.to_markdown(index=False))
        fh.write("\n\n## Parse-failure rate\n\n" + pf.reset_index().round(4).to_markdown(index=False))
        if len(agree):
            fh.write("\n\n## Overlap agreement\n\n" + agree.round(4).to_markdown(index=False))
        if len(sa):
            fh.write("\n\n## Shard accuracy (combined signal)\n\n" + sa.round(4).to_markdown(index=False))
        fh.write(f"\n\nMerged owner rows: {len(own)} → merged_results.jsonl\n")
    print(ch.to_string(index=False))
    return 1 if (ch["status"] == "FAIL").any() else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=5, help="repeated samples per order besides greedy")
    ap.add_argument("--max-parse-fail", type=float, default=0.02)
    ap.add_argument("--min-agree", type=float, default=0.95)
    ap.add_argument("--runner-from-file", action="store_true",
                    help="use each file's folder name as runner when the runner field is missing")
    a = ap.parse_args(argv)
    sys.exit(run(a.results, a.shards, a.pairs, a.out, a.samples, a.max_parse_fail, a.min_agree,
                 a.runner_from_file))


if __name__ == "__main__":
    main()
