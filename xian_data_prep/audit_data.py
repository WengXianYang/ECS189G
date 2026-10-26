"""Independently check teammate pair files against the raw datasets (stdlib only)."""

import argparse
import csv
import gzip
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from split_data import BASE, csv_bytes, json_bytes, normalize, read_csv, sha256


def close_exported(value, expected, decimals):
    # Teammate outputs round CTR/p to 6 decimals, z/relative effects to 4.
    return math.isclose(float(value), expected, rel_tol=0, abs_tol=0.5 * 10 ** (-decimals) + 1e-12)


def audit_upworthy(raw_path, pairs):
    _, raw = read_csv(raw_path)
    tests = defaultdict(list)
    for row in raw:
        tests[row["clickability_test_id"]].append(row)
    clean = {tid: rows for tid, rows in tests.items()
             if not any("2013-06-25" <= r["created_at"][:10] <= "2014-01-10" for r in rows)}
    counts = {"raw_packages": len(raw), "raw_tests": len(tests),
              "after_randomization_filter_tests": len(clean),
              "after_randomization_filter_packages": sum(len(p) for p in clean.values())}
    candidates = Counter()
    eligible_tests = set()
    isolation_tests = set()
    for tid, packages in clean.items():
        for a, b in combinations(packages, 2):
            if (a["headline"].strip().lower() == b["headline"].strip().lower()
                    or not a["eyecatcher_id"] or a["eyecatcher_id"] != b["eyecatcher_id"]
                    or a["excerpt"] != b["excerpt"]):
                continue
            isolation_tests.add(tid)
            candidates["candidate_pairs_before_impression_filter"] += 1
            na, nb = int(a["impressions"]), int(b["impressions"])
            if min(na, nb) < 500:
                continue
            candidates["candidate_pairs_after_min500_before_ties"] += 1
            if int(a["clicks"]) * nb == int(b["clicks"]) * na:
                candidates["exact_ctr_tie_candidates"] += 1
                continue
            candidates["eligible_non_tied_candidate_pairs"] += 1
            eligible_tests.add(tid)
    counts.update(candidates)
    counts["tests_with_isolated_pairs"] = len(isolation_tests)
    counts["tests_with_eligible_non_tied_pairs"] = len(eligible_tests)
    counts["selected_pairs"] = len(pairs)

    failures, checks = [], []
    for pair in sorted(pairs, key=lambda p: p["test_id"]):
        tid = pair["test_id"]
        errors = []
        found = {}
        for side in ("A", "B"):
            matches = [r for r in clean.get(tid, [])
                       if r["headline"].strip() == pair[f"headline_{side}"]
                       and r["eyecatcher_id"] == pair["eyecatcher_id"] and r["excerpt"] == pair["excerpt"]]
            # Match identity and then independently compare the exported measurements.
            matches = [r for r in matches if int(r["impressions"]) == int(pair[f"impressions_{side}"])
                       and int(r["clicks"]) == int(pair[f"clicks_{side}"])]
            if len(matches) != 1:
                errors.append(f"{side}: expected one matching raw package, found {len(matches)}")
            else:
                found[side] = matches[0]
        if len(found) == 2:
            a, b = found["A"], found["B"]
            ca, cb, na, nb = int(a["clicks"]), int(b["clicks"]), int(a["impressions"]), int(b["impressions"])
            ctr_a, ctr_b = ca / na, cb / nb
            difference = ca * nb - cb * na
            winner = "A" if difference > 0 else "B"
            pooled = (ca + cb) / (na + nb)
            se = math.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
            z = (ctr_a - ctr_b) / se if se else 0.0
            p = math.erfc(abs(z) / math.sqrt(2))
            expected = {"ctr_A": (ctr_a, 6), "ctr_B": (ctr_b, 6), "delta_ctr": (ctr_a - ctr_b, 6),
                        "abs_relative_diff": (abs(ctr_a - ctr_b) / max(ctr_a, ctr_b), 4),
                        "z_score": (z, 4), "p_value": (p, 6)}
            errors.extend(k + " differs from raw recomputation" for k, (value, dp) in expected.items()
                          if not close_exported(pair[k], value, dp))
            if not difference or pair["true_winner"] != winner:
                errors.append("winner differs from exact raw CTR comparison")
            if (pair["is_significant"].lower() == "true") != (p < 0.05):
                errors.append("significance differs from unrounded p < .05")
            if min(na, nb) < 500 or a["headline"].strip().lower() == b["headline"].strip().lower():
                errors.append("pair eligibility failed")
            if pair["created_at"] != a["created_at"]:
                errors.append("created_at differs from raw package A")
            checks.append({"test_id": tid, "raw_package_A_index": a[""], "raw_package_B_index": b[""],
                           "clicks_A_raw": ca, "impressions_A_raw": na, "ctr_A_recomputed": repr(ctr_a),
                           "ctr_A_exported": pair["ctr_A"], "clicks_B_raw": cb, "impressions_B_raw": nb,
                           "ctr_B_recomputed": repr(ctr_b), "ctr_B_exported": pair["ctr_B"],
                           "winner_recomputed": winner, "winner_exported": pair["true_winner"],
                           "passed": not errors})
        if errors:
            failures.append({"test_id": tid, "errors": errors})
    pair_ids = [p["test_id"] for p in pairs]
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != eligible_tests:
        failures.append({"errors": ["Selected test IDs do not cover eligible tests exactly once"]})
    # Fixed SHA ranking selects a stable sample without importing the producer's selection code.
    sample = sorted(checks, key=lambda r: sha256(f"42:raw-check:{r['test_id']}".encode()))[:100]
    report = {"counts": counts, "pairs_checked_against_raw": len(checks), "failures": failures,
              "sample_size": len(sample), "sample_rule": "first 100 by SHA256('42:raw-check:' + test_id)",
              "minimum_selected_impressions": min(int(p[f"impressions_{s}"]) for p in pairs for s in ("A", "B")),
              "pairs_with_identical_group_normalization": sum(normalize(p["headline_A"]) == normalize(p["headline_B"]) for p in pairs),
              "ctr_precision": "6 decimals; tolerance 0.0000005 plus 1e-12; winner uses exact integer cross-products"}
    return report, sample


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def audit_reddit(raw_path, pairs, supplied_counts):
    with gzip.open(raw_path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [c.strip().lstrip("#") for c in next(reader)]
        raw = []
        malformed = 0
        for values in reader:
            if len(values) > len(header):
                malformed += 1
                continue
            # pandas fills short records with missing values; they are removed in cleaning.
            values += [""] * (len(header) - len(values))
            raw.append(dict(zip(header, values)))
    counts = {"raw_records": len(raw) + malformed, "raw_rows": len(raw), "malformed_rows_skipped": malformed,
              "raw_images": len({r["image_id"] for r in raw})}
    clean = [r for r in raw if r["title"].strip().lower() not in {"", "nan", "[deleted]", "[removed]"}
             and r["subreddit"].strip().lower() not in {"", "nan"}
             and r["image_id"].strip() and number(r["unixtime"]) is not None and number(r["score"]) is not None]
    counts["after_missing_filter"] = len(clean)
    unique = {}
    for row in clean:
        unique.setdefault(row["reddit_id"], row)
    counts["after_dedup"] = len(unique)
    by_id = {key: r for key, r in unique.items() if (number(r["total_votes"]) or 0) >= 1}
    counts["after_min_votes"] = len(by_id)
    failures, violations = [], []
    for pair in pairs:
        errors, found = [], {}
        for side in ("a", "b"):
            raw_post = by_id.get(pair[f"reddit_id_{side}"])
            if raw_post is None:
                errors.append(side + ": raw submission absent after cleaning")
                continue
            found[side] = raw_post
            text_fields = {"image_id": pair["image_id"], "subreddit": pair["subreddit"], "title": pair[f"title_{side}"]}
            for column, exported in text_fields.items():
                observed = raw_post[column].strip().lower() if column == "subreddit" else raw_post[column].strip()
                if observed != exported:
                    errors.append(side + ": " + column + " differs from raw")
            for column, exported_column in (("score", "score"), ("total_votes", "votes"), ("number_of_comments", "comments"), ("unixtime", "time")):
                if number(raw_post[column]) != number(pair[f"{exported_column}_{side}"]):
                    errors.append(side + ": " + column + " differs from raw")
        if len(found) == 2:
            a, b = found["a"], found["b"]
            score_a, score_b = float(a["score"]), float(b["score"])
            ta, tb = int(a["unixtime"]), int(b["unixtime"])
            gap = abs(ta - tb)
            first = "A" if ta < tb else "B" if tb < ta else "simultaneous"
            if score_a == score_b or pair["winner"] != ("A" if score_a > score_b else "B"):
                errors.append("winner/score tie check failed")
            if normalize(a["title"]) == normalize(b["title"]):
                errors.append("identical normalized titles")
            if pair["first_posted"] != first or not math.isclose(float(pair["gap_hours"]), gap / 3600, abs_tol=1e-9):
                errors.append("posting order/time gap check failed")
            if float(pair["abs_diff"]) != abs(score_a - score_b):
                errors.append("abs_diff check failed")
            if gap > 30 * 86400:
                violations.append({"pair_id": pair["pair_id"], "image_id": pair["image_id"],
                                   "time_gap_seconds": gap, "time_gap_days": gap / 86400,
                                   "reason": "exceeds rules_spec.md 30-day window"})
        if errors:
            failures.append({"pair_id": pair["pair_id"], "errors": errors})
    if len({p["image_id"] for p in pairs}) != len(pairs):
        failures.append({"errors": ["Duplicate images in supplied sample"]})
    for key in ("raw_rows", "raw_images", "after_missing_filter", "after_dedup", "after_min_votes"):
        if counts[key] != supplied_counts[key]:
            failures.append({"errors": [f"Raw cleaning count {key} differs from teammate count"]})
    return {"independently_recomputed_counts": counts, "pairs_checked_against_raw": len(pairs), "failures": failures,
            "supplied_sample_pairs": len(pairs), "pairs_within_30_days": len(pairs) - len(violations),
            "pairs_outside_30_days": len(violations), "supplied_counts": supplied_counts,
            "decision": "Keep supplied 500-pair sample unchanged, per Xian's explicit instruction; report window mismatch.",
            "eligibility_status": "Not fully compliant with the 30-day rule; not a newly locked Reddit test set."}, violations


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upworthy-raw", required=True, type=Path)
    parser.add_argument("--reddit-raw", required=True, type=Path)
    args = parser.parse_args()
    _, upworthy = read_csv(BASE / "inputs/upworthy_pairs.csv")
    _, reddit = read_csv(BASE / "inputs/reddit_pairs.csv")
    supplied = json.loads((BASE / "inputs/reddit_counts.json").read_text(encoding="utf-8"))
    uw, sample = audit_upworthy(args.upworthy_raw, upworthy)
    rd, violations = audit_reddit(args.reddit_raw, reddit, supplied)
    if supplied["sha256_reddit_pairs"] != sha256((BASE / "inputs/reddit_pairs.csv").read_bytes()):
        rd["failures"].append({"errors": ["Supplied Reddit SHA-256 differs from sample bytes"]})
    summary = json.loads((BASE / "outputs/split_summary.json").read_text())
    report = {"upworthy": uw, "reddit": rd,
              "raw_sha256": {"upworthy": sha256(args.upworthy_raw.read_bytes()), "reddit_gzip": sha256(args.reddit_raw.read_bytes())},
              "pair_inputs_sha256": {name: sha256((BASE / "inputs" / name).read_bytes())
                                     for name in ("upworthy_pairs.csv", "reddit_pairs.csv")},
              "integrity_status": "PASS" if not uw["failures"] and not rd["failures"] else "FAIL",
              "rule_compliance_status": "Upworthy passes; Reddit has a documented 30-day-window mismatch"}
    counts = uw["counts"]
    rows = []

    def add(dataset, stage, unit, count, evidence):
        rows.append(dict(dataset=dataset, stage=stage, unit=unit, count=count, evidence=evidence))

    for stage, unit, count in [
        ("raw", "tests", counts["raw_tests"]), ("raw", "packages", counts["raw_packages"]),
        ("after_randomization_filter", "tests", counts["after_randomization_filter_tests"]),
        ("after_randomization_filter", "packages", counts["after_randomization_filter_packages"]),
        ("same_image_excerpt_distinct_headlines", "candidate_pairs", counts["candidate_pairs_before_impression_filter"]),
        ("after_min500_before_ties", "candidate_pairs", counts["candidate_pairs_after_min500_before_ties"]),
        ("after_exact_tie_exclusion", "candidate_pairs", counts["eligible_non_tied_candidate_pairs"]),
        ("one_pair_per_eligible_test", "pairs", counts["selected_pairs"]),
        ("dev", "pairs", summary["splits"]["dev"]["pairs"]),
        ("locked_test", "pairs", summary["splits"]["test"]["pairs"]),
    ]:
        add("Upworthy", stage, unit, count, "independently recomputed")
    for stage, count in rd["independently_recomputed_counts"].items():
        add("Reddit", stage, "rows" if stage != "raw_images" else "images", count, "independently recomputed")
    for stage in ("image_subreddit_groups_with_pair", "images_with_pair", "pairs_after_tie_drop"):
        add("Reddit", stage, "groups" if stage == "image_subreddit_groups_with_pair" else "pairs", supplied[stage], "teammate-reported; sampling stages not independently rebuilt")
    for stage in ("supplied_sample_pairs", "pairs_within_30_days", "pairs_outside_30_days"):
        add("Reddit", stage, "pairs", rd[stage], "independently recomputed")
    out = BASE / "reports"
    out.mkdir(exist_ok=True)
    (out / "audit_report.json").write_bytes(json_bytes(report))
    (out / "upworthy_raw_check_sample.csv").write_bytes(csv_bytes(sample, list(sample[0])))
    (out / "reddit_window_violations.csv").write_bytes(csv_bytes(violations, ["pair_id", "image_id", "time_gap_seconds", "time_gap_days", "reason"]))
    (out / "eligible_pair_counts.csv").write_bytes(csv_bytes(rows, ["dataset", "stage", "unit", "count", "evidence"]))
    print(f"Raw-data integrity: {report['integrity_status']}; checked {uw['pairs_checked_against_raw']} Upworthy and {rd['pairs_checked_against_raw']} Reddit pairs.")
    print(f"Reddit 30-day rule: {rd['pairs_within_30_days']} compliant, {rd['pairs_outside_30_days']} outside window; sample unchanged.")
    if report["integrity_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
