# Qwen 3 14B dev pilot (379 scoring pairs)

Run: Gaeun's RTX 5060 Ti 16 GB, Ollama, frozen protocol `headline-prediction-v1`, Nov 6–7 2026.
12,128 calls, 0 missing, 0 parse failures. Rebuild:
```
python tools/merge_check.py --results "results/dev/qwen/*.jsonl" --shards data/shards/prompt_dev_shards.csv --pairs data/prompt_dev.csv --out reports/dev_qwen/merge
python tools/metrics.py --pairs data/prompt_dev.csv --results reports/dev_qwen/merge/merged_results.jsonl --out reports/dev_qwen/metrics
```

| Method | Accuracy (stated) | 95% CI | AUROC (confidence vs correct) | ECE (stated) | Order consistency | Picked first | Speed |
|---|---|---|---|---|---|---|---|
| zero-shot | 53.0% | 48.0–58.0% | 0.555 | 0.151 | 75.7% | 51.3% | 0.75 calls/s |
| few-shot | 53.6% | 48.5–58.5% | 0.565 | 0.070 | 63.6% | 47.1% | ~0.8 calls/s |
| reasoning | 54.9% | 49.9–59.8% | 0.535 | 0.085 | 73.9% | 56.2% | 0.07 calls/s |

Takeaways
- Accuracy is barely above chance for every method; all 95% CIs reach 50%.
- Confidence carries little signal (AUROC 0.53–0.58). Stated confidence clusters at 75%.
- Reasoning is not more accurate than zero-shot but is ~11× slower (about 12 h vs 1.6 h on dev).
- Few-shot lowers order consistency (64%), i.e. answers flip more when the order is swapped.
- merge_check flags reasoning shard D1 (50.0%) vs D2 (59.2%) as outside each other's CI; with ~190 pairs per shard this is plausibly noise, not a pipeline issue (single runner, same settings).

Test-phase cost estimate on this GPU (1,543 Upworthy + 500 Reddit = 2,043 pairs)
| Plan | Calls | Time |
|---|---|---|
| zero-shot + few-shot, full sampling | 49,032 | ~18 h |
| + reasoning, greedy only (both orders) | +4,086 | +~16 h |
| + reasoning, full sampling (3 repeats) | +16,344 | +~65 h |
