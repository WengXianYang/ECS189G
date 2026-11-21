# Qwen 3 14B test-phase run (zero-shot + few-shot)

Run on Gaeun's RTX 5060 Ti 16 GB via Ollama, Nov 19–20, 2026, with the frozen protocol
`headline-prediction-v1` (protocol SHA-256 `d0162314…`, model digest `bdbd181c33f2…`).

| Dataset | Pairs | Calls | Missing | Parse failures | Input SHA-256 |
|---|---|---|---|---|---|
| Upworthy test | 1,543 | 37,032 | 0 | 0 | `cf4bad68…` (matches TEST_LOCK) |
| Reddit | 500 | 12,000 | 0 | 0 | `8a444425…` |

- Methods: zero-shot and few-shot, both orders, greedy + 5 samples each.
- **Reasoning not run yet**, pending the team decision on option B (greedy only). If it isn't run, report it as a limitation.
- **No accuracy has been computed or viewed.** The coverage checks ran without the label file
  (`merge_check.py` without `--pairs`). Score only at the final evaluation, together with the API models.

Rebuild the checks:
```
python tools/merge_check.py --results "results/test/qwen/*.jsonl" --shards data/shards/upworthy_test_shards.csv --out reports/test_qwen/merge
python tools/merge_check.py --results "results/reddit/qwen/*.jsonl" --shards data/shards/reddit_shards.csv --out reports/reddit_qwen/merge
```
