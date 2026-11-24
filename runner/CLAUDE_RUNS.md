# Claude API runs (owner: Gaeun)

Paid from Gaeun's Claude Max monthly API credit ($100; claimable from Nov 28, 2026). Run from the
repo root with `ANTHROPIC_API_KEY` set. Gaeun is the only runner for Claude models, so every pair
is run (no `--only-shard`), the same as for Qwen.

## Models
| Role | Model ID | Price in / out per 1M tokens | Scope |
|---|---|---|---|
| Small | `claude-haiku-5-5` | $0.10 / $0.50 | all three methods, full sampling (same as the frozen protocol) |
| Most capable | `claude-fable-5-2` | $8 / $40 | zero-shot + few-shot, greedy only (budget) |

## Deviations from the frozen protocol (report these)
- **Temperature is not sent.** Claude 5.x models return HTTP 400 for any non-default temperature, so
  the API default sampling is used. Rows keep the protocol label (`temperature` 0 for `sample_idx` 0,
  0.7 for repeats) and record `temperature_sent: false`.
- **Haiku zero/few-shot run with thinking disabled** (`--thinking disabled`), matching Qwen's
  `think: false`. Haiku reasoning uses adaptive thinking at effort `medium`.
- **Fable always thinks** (thinking can't be disabled on Fable 5.x). Zero/few-shot use adaptive thinking
  at the lowest effort (`low`), so Fable's "zero-shot" includes some hidden reasoning.
- `max_tokens` is raised where thinking is on (thinking counts toward it): 256 for Haiku without
  thinking, 4096 for Fable zero/few, 8192 for Haiku reasoning.

## Order of runs
```
export ANTHROPIC_API_KEY=...        # Console key; set a $100 workspace spend limit
R="python runner/run.py --provider anthropic --runner gaeun --workers 4"
DEV="--pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv --out-dir results/dev/claude"
TEST="--pairs xian_data_prep/outputs/upworthy_test.csv --shards data/shards/upworthy_test_shards.csv --out-dir results/test/claude"
RED="--pairs reddit/reddit_pairs.csv --shards data/shards/reddit_shards.csv --out-dir results/reddit/claude"
HZ="--model claude-haiku-5-5 --methods zero_shot few_shot --thinking disabled --max-tokens 256"
HR="--model claude-haiku-5-5 --methods reasoning --thinking adaptive --reasoning-effort medium --max-tokens 8192"
FZ="--model claude-fable-5-2 --methods zero_shot few_shot --thinking adaptive --reasoning-effort low --max-tokens 4096 --greedy-only"

# 1. smoke tests (1 call each)
$R $DEV $HZ --smoke ; $R $DEV $FZ --smoke
# 2. Haiku dev (12,128 calls), then check cost
$R $DEV $HZ ; $R $DEV $HR
python tools/api_cost.py "results/dev/claude/*haiku*.jsonl" --project-pairs 2422
# 3. Fable dev pilot: 20 pairs, then project the full greedy-only cost
$R $DEV $FZ --limit 20
python tools/api_cost.py "results/dev/claude/*fable*.jsonl" --project-pairs 2422 --greedy-only
# 4. Haiku test + Reddit
$R $TEST $HZ ; $R $TEST $HR ; $R $RED $HZ ; $R $RED $HR
# 5. Fable dev (rest), test, Reddit, only if the projection fits the remaining credit;
#    otherwise switch FZ to claude-opus-5-5 (half the price; also always thinks)
$R $DEV $FZ ; $R $TEST $FZ ; $R $RED $FZ
```
Every command is resume-safe: re-running it skips finished calls. After each stage:
`python tools/merge_check.py --results "results/<split>/claude/*.jsonl" --shards <shard file> --out reports/<split>_claude/merge`
(add `--pairs` only for dev; test and Reddit are scored at the final evaluation).
