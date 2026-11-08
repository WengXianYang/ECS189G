# Analysis tools (owner: Gaeun)

Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `tabulate`. All scripts were tested on
synthetic results (`make_synthetic.py`) and the current Upworthy split; `test_tools.py` has hand-checked cases.

| Script | What it does |
|---|---|
| `shards.py` | Stratified equal shards + overlap set every runner also runs |
| `merge_check.py` | Checks all runners' results (schema, duplicates, parse failures, coverage, overlap agreement, shard CIs) and writes one merged file |
| `metrics.py` | Accuracy, Brier, ECE, AUROC, risk-coverage/AURC, position bias, skipping policy and CTR regret; tables + plots |
| `make_synthetic.py` | Fake results in the shared schema, for testing |
| `common.py` | Shared loaders |

## Result schema (run.py must write this)
One JSON object per line, file `results_{shard}_{model}_{method}.jsonl`:

| Field | Meaning |
|---|---|
| `pair_id` | `test_id` (Upworthy) or `pair_id` (Reddit) |
| `shard` | owner shard from the shard file (`A`/`B`/`C`, `D1`/`D2`); use `O` for an overlap copy of someone else's pair |
| `model`, `method` | exact model string; `zero_shot` / `few_shot` / `reasoning` |
| `order` | `AB` = headline A shown first, `BA` = B shown first |
| `sample_idx` | 0 = greedy call (temperature 0); 1..N = repeated samples |
| `temperature` | recommended (if absent, `sample_idx == 0` is treated as greedy) |
| `choice` | **original** label `A` or `B` (map back from the shown position); `null` if unparseable |
| `stated_conf` | confidence the choice is right, 0.5–1 (0–100 also accepted) |
| `raw_output`, `latency`, `timestamp` | as returned |
| `runner` | `gaeun` / `yurun` / `xian` (needed for the overlap check) |

## Typical flow
```
# 1. shards (after the split is final)
python tools/shards.py --pairs xian_data_prep/outputs/upworthy_test.csv --labels A B C --overlap 150 --out data/shards/upworthy_test_shards.csv
python tools/shards.py --pairs reddit/reddit_pairs.csv --labels A B C --overlap 0 --out data/shards/reddit_shards.csv
python tools/shards.py --pairs xian_data_prep/outputs/upworthy_dev.csv --labels D1 D2 --overlap 60 --out data/shards/upworthy_dev_shards.csv

# 2. check and merge everyone's results
python tools/merge_check.py --results "results/test/*/*.jsonl" --shards data/shards/upworthy_test_shards.csv \
    --pairs xian_data_prep/outputs/upworthy_test.csv --out reports/merge_test

# 3. metrics on the merged file (thresholds chosen on dev)
python tools/metrics.py --pairs xian_data_prep/outputs/upworthy_test.csv --results reports/merge_test/merged_results.jsonl \
    --dev-pairs xian_data_prep/outputs/upworthy_dev.csv --dev-results reports/merge_dev/merged_results.jsonl \
    --baseline features=baselines/features_test.csv --dev-baseline features=baselines/features_dev.csv \
    --out reports/metrics_test
```
Baselines are CSVs with `pair_id, p_A` (probability that A wins).

## Definitions
- **Signals:** `stated` = mean stated confidence of the two greedy calls (as P(A wins));
  `consistency` = share of all valid calls (both orders, all samples) choosing A; `combined` = their mean.
  Prediction = A if P(A) > 0.5; an exact 0.5 falls back to the greedy AB choice.
- **Skipping policy:** ship the predicted winner without testing when confidence ≥ τ, else run the test.
  τ = lowest threshold whose **dev** selective accuracy reaches `--target-acc` (default 0.80, at least 20 pairs).
  CTR regret of a shipped pair = best CTR − CTR of the shipped headline; a tested pair has 0 regret.
- **Merge rule:** only owner-shard rows go into `merged_results.jsonl`; overlap copies are only used
  for the agreement check (FAIL below 95%).
