# Upworthy split v2

This version repairs the headline-only split from the `Eli` branch, merged into
main in `0738207` after PR #2 was closed without merging. The original split's
source commit is `d4e80650af0c3a3eeb3736a0ac4a0c3e67ebcfb8`. The original
outputs, lock, source and documentation remain in `archive/headline-v1/`.
Use the current `outputs/` files for all new work.

| Split | Pairs | Winner A | Winner B | Significant | Components |
|---|---:|---:|---:|---:|---:|
| Development | 386 | 188 | 198 | 111 | 359 |
| Locked test | 1,543 | 752 | 791 | 440 | 1,429 |

The new test-file SHA-256 is:

```text
cf4bad68874fb42c7f01dfb87651fa95b0dc57fc17ec058286141a846a8417d3
```

## Grouping and leakage

Pairs belong to the same connected component when they share **either** normalized
headline or the same nonempty image ID, including transitive links through other
pairs. A match on either key joins the pairs. Headline normalization remains the specified
`re.sub(r'[^a-z0-9]', '', text.lower())`. Image IDs are opaque strings with only
surrounding whitespace stripped; punctuation and case are preserved.

Excerpts are not grouping keys or model inputs. Boilerplate such as Upworthy's
generic tagline occurs in hundreds of unrelated pairs; grouping every shared
excerpt would create a large, semantically uninformative component. The current
leakage report records six shared nonempty excerpt strings as an informational
count. This split guarantees headline/image separation; excerpt separation is
outside its scope. Any future use of excerpts as model inputs requires a new
review of the grouping policy.

V1 had 30 shared image IDs affecting 32 test pairs. Its archived headline-only
leakage report did not check images.

The 20%/80% allocation, seed 42, largest-remainder winner/significance quotas,
hash-based ranking and exact CTR label validation are unchanged. The grouping
change produces 1,788 components (1,662 singletons, 113 pairs, 11 triples and two
groups of four). Group IDs and split membership differ from v1. No performance
results were used to choose the new split. Never combine results from the two
versions.

## Verification

Run from the repository root with Python 3.10+ and no additional packages:

```console
python xian_data_prep/split_data.py
python -m unittest discover -s xian_data_prep -p test_split_data.py -v
```

`TEST_LOCK.json` records input and output hashes. Regeneration compares every
existing output and the lock before writing anything; mismatches fail without
refreshing the lock. Missing matching outputs can be recreated. To reproduce in
a separate directory while checking the same lock:

```console
python xian_data_prep/split_data.py --out-dir xian_data_prep/reproduction_check
```

Seven regression checks cover transitive headline/image grouping, opaque image
IDs, missing-image rejection, real-data quotas and overlap, preservation of
all source rows and values, input-order independence, and refusal to overwrite
a changed lock or file. Independent
teammate confirmations remain pending in `TEST_CONFIRMATIONS.md`.

## Inputs and previous raw-data audit

The upstream pair snapshots are unchanged. The original independent raw-record
audit and its reports are retained from the merged `Eli` branch; its raw-agreement findings
are separate from the new split. The 100-pair audit sample may contain test
records and must not be used for prompt tuning or few-shot selection.

The supplied Reddit sample remains unchanged: 308 of 500 pairs exceed the
team's 30-day rule and 192 satisfy it. Reddit remains a separate observational
domain, with no new dev/test allocation. See `inputs/README_reddit.md` and
`reports/reddit_window_violations.csv`.

To reproduce the independent raw audit with the local raw files:

```console
python xian_data_prep/audit_data.py --upworthy-raw ../../tmp_hdr/exploratory-packages.csv --reddit-raw ../../tmp_hdr/redditSubmissions.csv.gz
```

Current split CSVs retain all 18 upstream columns and append `content_group_id`.
`outputs/content_groups.csv` lists component assignments;
`outputs/split_summary.json` records quotas and counts;
`outputs/leakage_report.json` records overlap checks. Input documentation is an
unaltered historical snapshot; this README and the v2 lock define the active
split policy.
