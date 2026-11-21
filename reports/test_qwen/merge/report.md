# Merge and sanity check

| check               | status   | detail                                                                           |
|:--------------------|:---------|:---------------------------------------------------------------------------------|
| 1 schema            | PASS     | 0 bad JSON lines, 0 rows missing required fields                                 |
| 2 duplicates        | PASS     | 0 duplicate rows, 0 keys with conflicting choices (exact duplicates are dropped) |
| 3 parse failures    | PASS     | worst model x method parse-failure rate 0.00% (limit 2%)                         |
| 4 coverage          | PASS     | 0 of 37032 expected owner calls missing (see missing_calls.csv)                  |
| 5 overlap agreement | WARN     | no overlap pairs run by 2+ runners yet                                           |
| 6 shard accuracy    | SKIP     | all shards inside each other's 95% CI                                            |

## Parse-failure rate

| model     | method    |   parse_fail_rate |
|:----------|:----------|------------------:|
| qwen3:14b | few_shot  |                 0 |
| qwen3:14b | zero_shot |                 0 |

Merged owner rows: 37032 → merged_results.jsonl
