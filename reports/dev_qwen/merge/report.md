# Merge and sanity check

| check               | status   | detail                                                                                                           |
|:--------------------|:---------|:-----------------------------------------------------------------------------------------------------------------|
| 1 schema            | PASS     | 0 bad JSON lines, 0 rows missing required fields                                                                 |
| 2 duplicates        | PASS     | 0 duplicate rows, 0 keys with conflicting choices (exact duplicates are dropped)                                 |
| 3 parse failures    | PASS     | worst model x method parse-failure rate 0.00% (limit 2%)                                                         |
| 4 coverage          | PASS     | 0 of 12128 expected owner calls missing (see missing_calls.csv)                                                  |
| 5 overlap agreement | WARN     | no overlap pairs run by 2+ runners yet                                                                           |
| 6 shard accuracy    | WARN     | qwen3:14b/reasoning: shard D1 0.500 outside shard D2 CI; qwen3:14b/reasoning: shard D2 0.592 outside shard D1 CI |

## Parse-failure rate

| model     | method    |   parse_fail_rate |
|:----------|:----------|------------------:|
| qwen3:14b | few_shot  |                 0 |
| qwen3:14b | reasoning |                 0 |
| qwen3:14b | zero_shot |                 0 |

## Shard accuracy (combined signal)

| model     | method    | shard   |   n |   accuracy |   ci_lo |   ci_hi |
|:----------|:----------|:--------|----:|-----------:|--------:|--------:|
| qwen3:14b | few_shot  | D1      | 188 |     0.516  |  0.4449 |  0.5864 |
| qwen3:14b | few_shot  | D2      | 191 |     0.5759 |  0.505  |  0.6438 |
| qwen3:14b | reasoning | D1      | 188 |     0.5    |  0.4292 |  0.5708 |
| qwen3:14b | reasoning | D2      | 191 |     0.5916 |  0.5208 |  0.6589 |
| qwen3:14b | zero_shot | D1      | 188 |     0.5053 |  0.4345 |  0.576  |
| qwen3:14b | zero_shot | D2      | 191 |     0.5602 |  0.4893 |  0.6287 |

Merged owner rows: 12128 → merged_results.jsonl
