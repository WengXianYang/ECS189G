# Xian's data split and independent validation

Prepared for Xian Yang Weng on October 25, 2026. Inputs are unchanged snapshots of
Yurun's Upworthy pairs and Gaeun's Reddit sample. Their original folders were not
modified. Both scripts use Python 3.10+ and the standard library, with no packages
to install. Local validation ran on Python 3.13.5.

## Results and status

| Dataset / split | Pairs | Winner A | Winner B | Significant (p < .05) |
|---|---:|---:|---:|---:|
| Upworthy, all eligible tests | 1,929 | 940 | 989 | 551 |
| Upworthy, development | 386 | 188 | 198 | 111 |
| Upworthy, locked test | 1,543 | 752 | 791 | 440 |
| Reddit, supplied secondary-domain sample | 500 | 255 | 245 | Not defined |

Upworthy has **zero normalized headline overlap**, zero shared test IDs, and zero
shared headline groups between dev and test. There is also zero normalized title
overlap between the Reddit sample and either Upworthy split.

The local Upworthy lock is created and reproduced. **Yurun's and Gaeun's independent
hash confirmations remain pending.** See [TEST_CONFIRMATIONS.md](TEST_CONFIRMATIONS.md).
No model evaluation was performed.

**Reddit rule mismatch:** 308 of the supplied 500 pairs (61.6%) exceed the 30-day
posting window in Yurun's rules specification; 192 satisfy that window. Per Xian's
explicit instruction, all 500 supplied pairs are retained unchanged. Do not describe
this sample as fully compliant with the 30-day rule. It remains a separate
observational evaluation domain; this work does not create a Reddit dev/test split
or declare the Reddit sample newly locked.

## Reproduce and verify the split

Run from the repository root:

```console
python xian_data_prep/split_data.py
```

Expected test-file SHA-256:

```text
765e5a02f916ca9dceb8b7b6525b2c3d04c4b41958c0753cec1ef49475849e60
```

The script verifies the snapshotted inputs, regenerates all output bytes in memory,
and compares them with [TEST_LOCK.json](TEST_LOCK.json). Existing matching files
are left untouched. Missing outputs can be recreated only if they match the lock.
Any changed input, generated content, or existing output causes a failure before
any write. There is no force-overwrite or unlock option.

To regenerate into a separate folder while still checking the committed lock:

```console
python xian_data_prep/split_data.py --out-dir xian_data_prep/reproduction_check
```

The shipped input snapshots allow verification on another computer without raw
datasets or the teammates' original directories. Git attributes preserve input
and locked-output bytes, including line endings, on Windows and Unix.

The lock is an integrity and workflow control. Keep the manifest and outputs in
version control, tune prompts and thresholds on dev only, and reserve test for
the planned final evaluation. An approved data-rule change needs an explicitly
reviewed new dataset version; do not replace this manifest to hide a mismatch.

## Split rules

The approximately 20% dev / 80% test contingency in `inputs/rules_spec.md` gives
386 dev and 1,543 test pairs, meeting the minimum of 1,500 test pairs. The upstream
specification's approximate 1,549 test figure is not the actual 80% allocation
for 1,929 rows.

1. Normalize headlines exactly as specified upstream:
   `re.sub(r'[^a-z0-9]', '', text.lower())`. This is deliberately ASCII based.
2. Connect a pair's two normalized headlines and group whole connected
   components, including transitive sharing through other pairs. Grouping only
   the two headlines concatenated together would miss leakage.
3. Stratify jointly by `true_winner` and `is_significant`. Compute integer dev
   quotas with the largest-remainder method, preserving the total of 386.
4. Rank groups by SHA-256 of seed 42 and a content-derived group ID. Consider
   multi-pair components for dev at a 20% hash threshold, subject to the quotas;
   fill each remaining quota from ranked singleton groups. Fail if quotas cannot
   be met while keeping groups intact. No outcome-based seed search is used.
5. Sort outputs by `test_id`; serialize UTF-8 CSV with LF line endings and
   canonical JSON. Input row order and Python hash randomization cannot change
   the generated output bytes.

There are 1,880 components: 1,834 singletons, 43 with two pairs, and three with
three pairs. Dev contains 374 groups; test contains 1,506 groups.

| Stratum | Total | Dev quota / actual | Test |
|---|---:|---:|---:|
| A wins, not significant | 662 | 132 | 530 |
| A wins, significant | 278 | 56 | 222 |
| B wins, not significant | 716 | 143 | 573 |
| B wins, significant | 273 | 55 | 218 |

Five supplied Upworthy pairs have identical headlines under the stronger grouping
normalization, although their stripped, lowercased strings differ. They remain
eligible under Yurun's pair rule, which tests stripped lowercase text, and their
headlines are grouped together. No new eligibility exclusion was introduced.

## Final count table

Machine-readable version: [reports/eligible_pair_counts.csv](reports/eligible_pair_counts.csv).

| Dataset | Stage | Tests / groups | Packages / pairs | Evidence |
|---|---|---:|---:|---|
| Upworthy | Raw exploratory | 4,873 tests | 22,666 packages | Recomputed from raw |
| Upworthy | After randomization-window exclusion | 3,823 tests | 18,388 packages | Recomputed from raw |
| Upworthy | Same test, image, excerpt; distinct headlines | 1,929 tests | 16,486 candidate pairs | Recomputed from raw |
| Upworthy | Both impressions >= 500 | 1,929 tests | 16,486 candidate pairs | Recomputed from raw |
| Upworthy | Exclude exact CTR ties | 1,929 tests | 16,485 candidate pairs | Recomputed from raw |
| Upworthy | One pair per eligible test | 1,929 tests | 1,929 selected pairs | Recomputed from raw |
| Upworthy | Dev | 374 headline groups | 386 pairs | Generated / verified |
| Upworthy | Locked test | 1,506 headline groups | 1,543 pairs | Generated / verified |
| Reddit | Raw, including 5 malformed records | 16,736 images | 132,308 records | Recomputed from raw |
| Reddit | After skipping extra-column records | 16,736 images | 132,303 rows | Recomputed from raw |
| Reddit | After missing/deleted-title filter | — | 132,302 rows | Recomputed from raw |
| Reddit | After duplicate reddit_id removal | — | 132,209 rows | Recomputed from raw |
| Reddit | After minimum-total-votes filter | — | 132,206 rows | Recomputed from raw |
| Reddit | Groups with distinct titles | 23,990 image/subreddit groups | — | Teammate-reported |
| Reddit | One pair per image before ties | 15,744 images | 15,744 pairs | Teammate-reported |
| Reddit | After score-tie exclusion | — | 15,411 pairs | Teammate-reported |
| Reddit | Supplied final sample, unchanged | 500 images | 500 pairs | All checked against raw |
| Reddit | Sample satisfying 30-day window | 192 images | 192 pairs | Recomputed from timestamps |
| Reddit | Sample exceeding 30-day window | 308 images | 308 pairs | Recomputed from timestamps |

The Upworthy tie is excluded **before** one-pair-per-test selection. It removes
one candidate, not a selected test. This clarifies the upstream README's statement
that one pair was dropped while still reporting 1,929 final pairs. Reddit's
intermediate sampling counts are attributed to Gaeun; that pipeline was not
rebuilt or resampled.

## Independent raw-data validation

All **1,929 Upworthy pairs** were independently matched to raw packages, exceeding
the requested sample check. The audit does not import Yurun's builder. It checks
test membership, image/excerpt equality, distinct headlines, the unreliable date
window, minimum impressions, clicks/impressions, CTRs, winner, effect sizes,
pooled z-test, p-value, significance flag, and creation time. Every eligible test
is present exactly once. There were **zero mismatches**. The minimum selected-arm
impression count was 758.

Exported CTRs have six decimal places, so comparisons allow half a unit in the
last exported decimal. Winners and CTR ties use exact integer cross-products,
never rounded CTR columns. Significance uses the unrounded two-sided p-value
below .05, matching the actual producer code. The upstream 1.96 z-cutoff is an
approximation; it causes no discrepancies in these pairs.

[reports/upworthy_raw_check_sample.csv](reports/upworthy_raw_check_sample.csv)
contains a fixed 100-pair comparison with raw package row IDs, counts,
recomputed/exported CTRs, and winner comparisons. Sample membership uses SHA-256
ranking with seed 42. It is an audit artifact; use the dev file for few-shot
examples and prompt development.

All **500 Reddit pairs** also match their raw submissions on image, subreddit,
title, score, votes, comments, timestamps, winner, score difference, posting order,
and time gap, with no duplicate sampled images or identical normalized title
pairs. Raw cleaning counts and the supplied sample hash match. Window violations
are listed in [reports/reddit_window_violations.csv](reports/reddit_window_violations.csv).

To rerun the audit, supply the original raw files. In Xian's current folder layout,
run from the repository root:

```console
python xian_data_prep/audit_data.py --upworthy-raw ../../tmp_hdr/exploratory-packages.csv --reddit-raw ../../tmp_hdr/redditSubmissions.csv.gz
```

On another computer, replace those paths with the local raw-file paths. Raw files
are not bundled here. Their SHA-256 values are recorded in
[reports/audit_report.json](reports/audit_report.json). Investigate changed raw
files before replacing that audit record. `integrity_status: PASS` means raw-record
agreement; `rule_compliance_status` separately reports the Reddit window mismatch.

Run regression checks from the repository root:

```console
python -m unittest discover -s xian_data_prep -p test_split_data.py -v
```

Four checks passed: transitive sharing, real-data split/stratification, identical
output bytes after input-row reordering, and refusal to overwrite after manifest
drift or file tampering. A separate regeneration also matched all output hashes.

## Files and columns

| File | Purpose |
|---|---|
| `inputs/upworthy_pairs.csv` | Byte-identical snapshot of Yurun's pairs |
| `inputs/reddit_pairs.csv` | Byte-identical snapshot of Gaeun's sample |
| `inputs/reddit_counts.json` | Gaeun's counts and sample hash |
| `inputs/rules_spec.md`, `inputs/README_*.md` | Unmodified teammate documentation |
| `outputs/upworthy_dev.csv` | Prompt/baseline development data |
| `outputs/upworthy_test.csv` | Locked Upworthy evaluation data |
| `outputs/headline_groups.csv` | Group ID, split, pair count |
| `outputs/split_summary.json` | Sizes, strata and group counts |
| `outputs/leakage_report.json` | Headline, group, test ID and cross-domain overlaps |
| `TEST_LOCK.json` | Input/output SHA-256 values |
| `reports/audit_report.json` | Independent raw checks and provenance hashes |
| `reports/eligible_pair_counts.csv` | Final count table with evidence labels |
| `reports/upworthy_raw_check_sample.csv` | 100 raw-package/CTR/winner comparisons |
| `reports/reddit_window_violations.csv` | Pair IDs and exact gaps for 308 violations |

Upworthy dev/test preserve all 18 upstream columns and values, adding only
`headline_group_id`, a content-derived component ID. See the input README for
upstream column definitions. The Reddit snapshot preserves all 20 columns.
Its `gap_hours` is checked against raw timestamps; the violation report includes
exact `time_gap_seconds` for pairs outside the window.

## Sources and attribution

Yurun Shi produced the Upworthy pairs and rules specification. Gaeun Yu produced
the Reddit pairs and counts. This folder adds Xian's split and independent audit.

The [Upworthy Research Archive](https://upworthy.natematias.com/) documents the
public exploratory subset. The local raw file predates the new reliability flag;
the audit applies the team's inclusive June 25, 2013–January 10, 2014 date filter.
The authors' [2024 update](https://upworthy.natematias.com/2024-06-upworthy-archive-update.html)
discourages causal research on that interval because of possible randomization
problems. The cache explanation is their hypothesis, not a conclusively
established mechanism.

The [SNAP Reddit dataset page](https://snap.stanford.edu/data/web-Reddit.html)
defines image IDs, timestamps and score (upvotes minus downvotes), and cites
Lakkaraju, McAuley and Leskovec (ICWSM 2013). Reddit is observational: matching
image and subreddit does not make title effects causal.
