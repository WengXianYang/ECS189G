# Metrics report

Pairs: `data/prompt_dev.csv`

## Summary

| model     | method    | signal      |   n |   accuracy |   acc_ci_lo |   acc_ci_hi |   acc_significant |   n_significant |   brier |    ece |   auroc |   aurc |   mean_conf |   mean_regret_all_shipped |
|:----------|:----------|:------------|----:|-----------:|------------:|------------:|------------------:|----------------:|--------:|-------:|--------:|-------:|------------:|--------------------------:|
| qwen3:14b | few_shot  | combined    | 379 |     0.5462 |      0.4958 |      0.5956 |            0.5514 |             107 |  0.279  | 0.1736 |  0.5528 | 0.4029 |      0.7047 |                    0.0019 |
| qwen3:14b | few_shot  | consistency | 379 |     0.5462 |      0.4958 |      0.5956 |            0.5701 |             107 |  0.3396 | 0.259  |  0.5442 | 0.4117 |      0.8052 |                    0.0019 |
| qwen3:14b | few_shot  | stated      | 379 |     0.5356 |      0.4853 |      0.5852 |            0.514  |             107 |  0.2503 | 0.0696 |  0.5647 | 0.4101 |      0.6052 |                    0.002  |
| qwen3:14b | reasoning | combined    | 379 |     0.5462 |      0.4958 |      0.5956 |            0.5327 |             107 |  0.2899 | 0.1927 |  0.5318 | 0.4402 |      0.7382 |                    0.0019 |
| qwen3:14b | reasoning | consistency | 379 |     0.5488 |      0.4985 |      0.5982 |            0.5421 |             107 |  0.3576 | 0.2991 |  0.5245 | 0.4372 |      0.848  |                    0.0019 |
| qwen3:14b | reasoning | stated      | 379 |     0.5488 |      0.4985 |      0.5982 |            0.514  |             107 |  0.2555 | 0.0854 |  0.5352 | 0.4336 |      0.6317 |                    0.0019 |
| qwen3:14b | zero_shot | combined    | 379 |     0.533  |      0.4827 |      0.5826 |            0.514  |             107 |  0.3041 | 0.246  |  0.5753 | 0.4138 |      0.7705 |                    0.0019 |
| qwen3:14b | zero_shot | consistency | 379 |     0.5435 |      0.4932 |      0.593  |            0.5327 |             107 |  0.3621 | 0.3175 |  0.5639 | 0.4028 |      0.861  |                    0.0019 |
| qwen3:14b | zero_shot | stated      | 379 |     0.5303 |      0.48   |      0.58   |            0.5421 |             107 |  0.2711 | 0.1513 |  0.555  | 0.4252 |      0.6816 |                    0.0019 |

## Position bias and order consistency (greedy calls)

| model     | method    |   pairs |   order_consistency |   picked_first_rate |
|:----------|:----------|--------:|--------------------:|--------------------:|
| qwen3:14b | few_shot  |     379 |              0.6359 |              0.471  |
| qwen3:14b | reasoning |     379 |              0.7388 |              0.562  |
| qwen3:14b | zero_shot |     379 |              0.7573 |              0.5132 |

Figures: reliability.png, risk_coverage.png
