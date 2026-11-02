#!/usr/bin/env python3
"""
build_upworthy_pairs.py
Author: Yurun Shi (ECS 189G)
Project: Can LLMs Predict A/B Test Results, and Do They Know When They're Wrong?

Tasks Addressed:
- Task 2: Apply 2024 update (filter unreliable randomization: 2013-06-25 to 2014-01-10).
- Task 3: Build Upworthy pairwise dataset (1 pair per test, same image & excerpt, seed=42).
- Output: upworthy_pairs.csv
"""

import csv
import math
import random
import sys
from collections import defaultdict

RAW_DATA_PATH = "upworthy-archive-datasets/upworthy-archive-exploratory-packages-03.12.2020.csv"
OUTPUT_PAIRS_PATH = "upworthy_pairs.csv"
RANDOM_SEED = 42
MIN_IMPRESSIONS = 500

# 2024 Cache Glitch window (unreliable randomization due to Cloudflare caching)
GLITCH_START = "2013-06-25"
GLITCH_END = "2014-01-10"

def calculate_two_proportion_z(c1, n1, c2, n2):
    """Computes two-proportion pooled z-score and two-tailed p-value."""
    ctr1 = c1 / n1
    ctr2 = c2 / n2
    p_pool = (c1 + c2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (ctr1 - ctr2) / se
    # Approximate two-tailed p-value from standard normal
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p_value

def main():
    print(f"=== Step 1: Loading raw exploratory packages from {RAW_DATA_PATH} ===")
    with open(RAW_DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    total_packages = len(raw_rows)
    tests = defaultdict(list)
    for r in raw_rows:
        tests[r["clickability_test_id"]].append(r)
    total_tests = len(tests)

    print(f"Loaded {total_packages} packages across {total_tests} unique tests.")

    print(f"\n=== Step 2: Applying 2024 Randomization Update ({GLITCH_START} to {GLITCH_END}) ===")
    clean_tests = {}
    dropped_glitch_tests = 0
    for tid, pkgs in tests.items():
        created_date = pkgs[0]["created_at"][:10]
        if GLITCH_START <= created_date <= GLITCH_END:
            dropped_glitch_tests += 1
        else:
            clean_tests[tid] = pkgs

    print(f"Dropped {dropped_glitch_tests} tests ({dropped_glitch_tests/total_tests*100:.1f}%) during cache glitch window.")
    print(f"Clean tests remaining: {len(clean_tests)}")

    print(f"\n=== Step 3: Finding Eligible Headline Pairs (Same image + same excerpt) ===")
    random.seed(RANDOM_SEED)
    eligible_tests = 0
    sampled_pairs = []

    for tid in sorted(clean_tests.keys()):
        pkgs = clean_tests[tid]
        candidate_pairs = []
        for i in range(len(pkgs)):
            for j in range(i + 1, len(pkgs)):
                p1, p2 = pkgs[i], pkgs[j]
                h1, h2 = p1["headline"].strip(), p2["headline"].strip()

                # Rule 1: Distinct headlines
                if h1.lower() == h2.lower():
                    continue

                # Rule 2: Identical eyecatcher (image) and non-empty
                if not p1["eyecatcher_id"] or p1["eyecatcher_id"] != p2["eyecatcher_id"]:
                    continue

                # Rule 3: Identical excerpt
                if p1["excerpt"] != p2["excerpt"]:
                    continue

                # Rule 4: Traffic validity & minimum impressions threshold
                imp1, imp2 = int(p1["impressions"]), int(p2["impressions"])
                if imp1 < MIN_IMPRESSIONS or imp2 < MIN_IMPRESSIONS:
                    continue

                # Rule 5: CTR calculation & tie check
                clk1, clk2 = int(p1["clicks"]), int(p2["clicks"])
                ctr1 = clk1 / imp1
                ctr2 = clk2 / imp2

                if ctr1 == ctr2:
                    continue  # Exclude exact ties

                candidate_pairs.append((p1, p2, clk1, imp1, ctr1, clk2, imp2, ctr2))

        if candidate_pairs:
            eligible_tests += 1
            # Pick 1 pair deterministically with fixed seed
            chosen = random.choice(candidate_pairs)
            p1, p2, clk1, imp1, ctr1, clk2, imp2, ctr2 = chosen

            # Position-neutral coin flip: decide whether p1 is A and p2 is B, or vice-versa
            if random.random() < 0.5:
                pkg_A, pkg_B = p1, p2
                c_A, n_A, ctr_A = clk1, imp1, ctr1
                c_B, n_B, ctr_B = clk2, imp2, ctr2
            else:
                pkg_A, pkg_B = p2, p1
                c_A, n_A, ctr_A = clk2, imp2, ctr2
                c_B, n_B, ctr_B = clk1, imp1, ctr1

            true_winner = "A" if ctr_A > ctr_B else "B"
            delta_ctr = ctr_A - ctr_B
            rel_diff = abs(ctr_A - ctr_B) / max(ctr_A, ctr_B) if max(ctr_A, ctr_B) > 0 else 0
            z, p_val = calculate_two_proportion_z(c_A, n_A, c_B, n_B)
            is_sig = (p_val < 0.05)

            sampled_pairs.append({
                "test_id": tid,
                "created_at": pkg_A["created_at"],
                "eyecatcher_id": pkg_A["eyecatcher_id"],
                "excerpt": pkg_A["excerpt"],
                "headline_A": pkg_A["headline"].strip(),
                "headline_B": pkg_B["headline"].strip(),
                "impressions_A": n_A,
                "clicks_A": c_A,
                "ctr_A": round(ctr_A, 6),
                "impressions_B": n_B,
                "clicks_B": c_B,
                "ctr_B": round(ctr_B, 6),
                "true_winner": true_winner,
                "delta_ctr": round(delta_ctr, 6),
                "abs_relative_diff": round(rel_diff, 4),
                "z_score": round(z, 4),
                "p_value": round(p_val, 6),
                "is_significant": is_sig
            })

    print(f"Eligible tests with valid pairs: {eligible_tests}")
    print(f"Total sampled pairs (1 per test): {len(sampled_pairs)}")
    sig_count = sum(1 for p in sampled_pairs if p["is_significant"])
    print(f"Pairs with statistically significant CTR difference (p < 0.05): {sig_count} ({sig_count/len(sampled_pairs)*100:.1f}%)")
    winner_A_count = sum(1 for p in sampled_pairs if p["true_winner"] == "A")
    print(f"Position check - Winner A: {winner_A_count} ({winner_A_count/len(sampled_pairs)*100:.1f}%), Winner B: {len(sampled_pairs)-winner_A_count} (Balanced coin-flip)")

    # Save to CSV
    fieldnames = list(sampled_pairs[0].keys())
    with open(OUTPUT_PAIRS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled_pairs)

    print(f"\nSuccessfully wrote {len(sampled_pairs)} pairs to {OUTPUT_PAIRS_PATH}")

    # Hand-check 20 pairs
    print("\n" + "="*80)
    print("=== Hand-Check Sample of 20 Pairs (Task 3 Requirement) ===")
    print("="*80)
    check_sample = random.sample(sampled_pairs, 20)
    for idx, p in enumerate(check_sample, 1):
        sig_str = "SIG (p<0.05)" if p["is_significant"] else "NON-SIG"
        print(f"[{idx:02d}] Test ID: {p['test_id']} ({sig_str})")
        print(f"     Headline A: \"{p['headline_A']}\"")
        print(f"     Headline B: \"{p['headline_B']}\"")
        print(f"     CTR A: {p['ctr_A']*100:.2f}% ({p['clicks_A']}/{p['impressions_A']}) | CTR B: {p['ctr_B']*100:.2f}% ({p['clicks_B']}/{p['impressions_B']})")
        print(f"     Winner: {p['true_winner']} (diff: {abs(p['delta_ctr'])*100:.2f}% pts, rel: {p['abs_relative_diff']*100:.1f}%)")
        print(f"     Excerpt: \"{p['excerpt'][:60]}...\" | Image: {p['eyecatcher_id'][:12]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()
