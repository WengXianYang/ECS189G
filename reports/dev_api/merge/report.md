# Merge and sanity check

| check               | status   | detail                                                                           |
|:--------------------|:---------|:---------------------------------------------------------------------------------|
| 1 schema            | PASS     | 0 bad JSON lines, 0 rows missing required fields                                 |
| 2 duplicates        | PASS     | 0 duplicate rows, 0 keys with conflicting choices (exact duplicates are dropped) |
| 3 parse failures    | PASS     | worst model x method parse-failure rate 0.00% (limit 2%)                         |
| 4 coverage          | PASS     | 0 of 12128 expected owner calls missing (see missing_calls.csv)                  |
| 5 overlap agreement | WARN     | no overlap pairs run by 2+ runners yet                                           |
| 6 shard accuracy    | PASS     | all shards inside each other's 95% CI                                            |

## Parse-failure rate

| model       | method    |   parse_fail_rate |
|:------------|:----------|------------------:|
| qwen3.6:35b | few_shot  |                 0 |
| qwen3.6:35b | zero_shot |                 0 |
| qwen3.8:27b | reasoning |                 0 |

## Shard accuracy (combined signal)

| model       | method    | shard   |   n |   accuracy |   ci_lo |   ci_hi |
|:------------|:----------|:--------|----:|-----------:|--------:|--------:|
| qwen3.6:35b | few_shot  | D1      | 188 |     0.5532 |  0.4818 |  0.6225 |
| qwen3.6:35b | few_shot  | D2      | 191 |     0.5969 |  0.526  |  0.6639 |
| qwen3.6:35b | zero_shot | D1      | 188 |     0.5372 |  0.4659 |  0.6071 |
| qwen3.6:35b | zero_shot | D2      | 191 |     0.5707 |  0.4998 |  0.6388 |
| qwen3.8:27b | reasoning | D1      | 188 |     0.5745 |  0.503  |  0.643  |
| qwen3.8:27b | reasoning | D2      | 191 |     0.6021 |  0.5313 |  0.6688 |

Merged owner rows: 12128 → merged_results.jsonl
