# Headline prediction protocol v1

The root `config.yaml`, these prompt files, and `few_shot.json` define one shared
protocol. `EXPERIMENT_LOCK.json` records their SHA-256 values, the selection code,
the runner, and the relevant data files. This version is frozen for the first
live dev pilot. Offline input checks have passed; accuracy and calibration have
not yet been measured because Ollama was unavailable on the preparation machine.

## Methods and confidence

`zero_shot.txt` asks for a direct prediction, `few_shot.txt` adds four fixed dev
demonstrations, and `reasoning.txt` asks for a considered prediction with Qwen's
native thinking enabled. All three use the same task definition, target text,
confidence question, and final answer format:

```json
{"choice": 1, "confidence": 70}
```

`choice` refers to the displayed position, so the runner maps it back to original
A/B labels after each AB/BA presentation. `confidence` is a probability in
percent that the selected option matches the original observed winner. It is
not a predicted CTR or an effect size. The demonstration outcomes carry no
invented confidence values. Target clicks, impressions, scores, labels, image
IDs, and excerpts never enter the prompt.

Upworthy uses CTR as the target. Reddit templates use score and subreddit as
context. Reddit few-shot runs use the same four Upworthy examples, explicitly
identified as cross-domain demonstrations. No Reddit labels were selected as
demonstrations or used for tuning. Upworthy remains the primary dev task.

## Frozen settings

Only the repository's configured model, `qwen3:14b` through Ollama, is included.
Its Q4_K_M artifact is pinned to the full manifest digest:

```text
sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8
```

The digest was computed from the public
[Ollama registry manifest](https://registry.ollama.ai/v2/library/qwen3/manifests/14b)
on November 4, 2026 and matches the prefix on the
[model page](https://ollama.com/library/qwen3:14b). Before inference, the runner
checks the installed digest through
[`/api/tags`](https://docs.ollama.com/api/tags) and records the Ollama version.
Thinking uses the API's
[`think` control](https://docs.ollama.com/capabilities/thinking).

| Setting | Zero-shot | Few-shot | Reasoning |
|---|---:|---:|---:|
| Native thinking | Off | Off | On |
| Greedy temperature / seed | 0 / 42 | 0 / 42 | 0 / 42 |
| Additional repeats per order | 5 | 5 | 3 |
| Repeat temperature | 0.7 | 0.7 | 0.7 |
| Repeat seeds | 43–47 | 43–47 | 43–45 |
| Orders | AB and BA | AB and BA | AB and BA |
| Output token budget | 64 | 64 | 2,048 |
| Context budget | 4,096 | 4,096 | 8,192 |

Generation settings come from `config.yaml`; incompatible command-line overrides
fail before inference. The file uses JSON syntax, a valid YAML subset, so the
standard-library-only runner needs no YAML package. API models can be added in a
new protocol version after their exact model IDs and access are known.

## Demonstration selection and dev scoring

`experiments/select_examples.py` selects one example from each dev
winner/significance stratum, giving two A winners, two B winners, two significant
and two nonsignificant comparisons. It uses seed 42 and SHA-256 ranking, with
distinct content components and fixed readability bounds. Examples are fixed
across targets, orders, and runs; no performance-based example search was done.

The original list of 51 flagged test IDs was not available. The exclusion policy
therefore covers a conservative superset: all archived v1 and current v2 test
pairs. A dev example is ineligible if it links transitively through a normalized
headline, image ID, or nonempty normalized excerpt to any of those test pairs.
Candidate headlines also must avoid near-duplicates of either test version.
Test outcomes are not used by the selection algorithm.

This leaves 32 candidate rows before the final four are selected.
`experiments/few_shot_selection.json` records the policy, IDs, source hashes,
balance, and exclusions. It does not claim to recover the exact original 51 IDs.

For all methods, dev scoring excludes the four demonstration-related dev rows
and three additional dev rows with near-duplicate current test headlines.
The near-duplicate rule uses ASCII-normalized text and the maximum of both
`difflib.SequenceMatcher` directions with `autojunk=False`, at a 0.90 threshold.
The resulting `data/prompt_dev.csv` contains 379 rows. Its derived shard file
preserves the original D1/D2 and overlap assignments for those rows. The original
386-row dev split, 1,543-row test split, lock and shard allocations are unchanged.

## Validate and run

From the repository root:

```console
python -m experiments.protocol
python qwen/run_qwen.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv --dry-run
python -m pytest experiments/test_protocol.py tools/test_tools.py -q
```

To reproduce the example selection and derived files, without overwriting drift:

```console
python experiments/select_examples.py
```

`experiments/dev_prompt_review.json` records successful rendering of all 379 dev
targets, all three methods, and both orders (2,274 prompts), as well as the
pending live-pilot status. It contains no measured model results.

On the GPU machine with the pinned model installed, start with a small dev pilot:

```console
python qwen/run_qwen.py --pairs data/prompt_dev.csv --shards data/shards/prompt_dev_shards.csv --limit 10 --runner xian --out-dir results/dev/qwen
```

This makes 320 calls with the frozen settings. The full dev run makes 12,128
calls: 4,548 zero-shot, 4,548 few-shot, and 3,032 reasoning. Use
`qwen/run_qwen_dev.bat` for the full run. A limited run can resume into the full
run using the same input file, result directory and protocol.

Use the derived dev pair and shard files when merging or analyzing these runs.
The existing merge tool accepts one repeat count per invocation, so merge
zero/few-shot with five repeats and reasoning separately with three:

```console
python tools/merge_check.py --results "results/dev/qwen/*_zero_shot.jsonl" "results/dev/qwen/*_few_shot.jsonl" --samples 5 --shards data/shards/prompt_dev_shards.csv --pairs data/prompt_dev.csv --out reports/merge_dev/standard
python tools/merge_check.py --results "results/dev/qwen/*_reasoning.jsonl" --samples 3 --shards data/shards/prompt_dev_shards.csv --pairs data/prompt_dev.csv --out reports/merge_dev/reasoning
```

Later metrics calls should use `--dev-pairs data/prompt_dev.csv` with both matching
merged dev result files. Thresholds and prompt revisions must use dev only. Do not
interpret the dev input review as meeting the November 4 checkpoint's requirement
for actual model results.

The runner saves protocol, prompt, input and model hashes plus the seed in each
result row. It refuses to resume results from a different protocol or input.
Invalid confidence values are recorded as missing rather than clipped; a corrupt
JSONL line stops resume and identifies the line requiring repair.

For a prompt or config revision after the live pilot, make an explicit new
protocol version, preserve the previous lock and results, regenerate the reviewed
artifacts, and freeze the new version. Do not replace a lock to hide drift or mix
results produced by different versions. No test inference was run during this work.
