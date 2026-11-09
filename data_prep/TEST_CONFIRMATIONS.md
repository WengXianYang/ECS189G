# Upworthy v2 test lock confirmations

File: `outputs/upworthy_test.csv` (1,543 pairs).

```text
cf4bad68874fb42c7f01dfb87651fa95b0dc57fc17ec058286141a846a8417d3
```

| Verification | Status | Evidence |
|---|---|---|
| Local generation and reproduction | PASS, November 1, 2026 | Seven regression checks pass; headline and image overlaps are zero |
| Yurun's independent reproduction | PASS, November 8, 2026 | Run by Yurun on macOS, Python 3.13: `split_data.py` printed "Verified: 386 dev / 1543 test pairs; zero headline/image overlap"; test SHA-256 `cf4bad68874fb42c7f01dfb87651fa95b0dc57fc17ec058286141a846a8417d3` matches |
| Gaeun's independent reproduction | PENDING | No result received |

Run `python xian_data_prep/split_data.py` from the repository root and compare the
full hash. Record teammates' confirmations only after receiving their results.
V1 confirmations and its superseded hash are archived in `archive/headline-v1/`.
No final test model evaluation is claimed by this record.
