# Upworthy test-set lock confirmations

Expected test file: `outputs/upworthy_test.csv` (1,543 pairs).

SHA-256:

```text
765e5a02f916ca9dceb8b7b6525b2c3d04c4b41958c0753cec1ef49475849e60
```

| Person / verification | Status | Evidence |
|---|---|---|
| Xian's local automated run | PASS, October 25, 2026 | Initial generation and separate regeneration match; all four regression checks pass |
| Yurun's independent run | PENDING | No confirmation received |
| Gaeun's independent run | PENDING | No confirmation received |

Each teammate should run this from the repository root, compare the full hash
above, and record their own confirmation with the date and hash:

```console
python xian_data_prep/split_data.py
```

An optional independent file hash check on Windows:

```powershell
Get-FileHash -Algorithm SHA256 xian_data_prep/outputs/upworthy_test.csv
```

The local lock is active; team-wide confirmation is pending. Do not mark another
person's run complete without their actual result. Keep prompts, examples and
threshold tuning on development data. Test labels are reserved for the planned
final evaluation. This record does not claim the test set has been evaluated or
that the Reddit sample satisfies the 30-day rule.
