# Upworthy v2 test lock confirmations

File: `outputs/upworthy_test.csv` (1,543 pairs).

```text
cf4bad68874fb42c7f01dfb87651fa95b0dc57fc17ec058286141a846a8417d3
```

| Verification | Status | Evidence |
|---|---|---|
| Local generation and reproduction | PASS, November 1, 2026 | Seven regression checks pass; headline and image overlaps are zero |
| Yurun's independent reproduction | PENDING | No result received |
| Gaeun's independent reproduction | PASS, November 3, 2026 | Run by Claude for Gaeun in a clean checkout of main (cdda04f), Linux, Python 3.13: `split_data.py` printed "Verified: 386 dev / 1543 test pairs; zero headline/image overlap"; test SHA-256 `cf4bad68874fb42c7f01dfb87651fa95b0dc57fc17ec058286141a846a8417d3` matches. Independent leakage check (own code): 0 shared test_id, normalized headline or image; 2 near-duplicate headline pairs remain (≥90% similar, see PR #4) |

Run `python xian_data_prep/split_data.py` from the repository root and compare the
full hash. Record teammates' confirmations only after receiving their results.
V1 confirmations and its superseded hash are archived in `archive/headline-v1/`.
No final test model evaluation is claimed by this record.
