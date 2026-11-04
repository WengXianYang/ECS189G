# Qwen 3 local runs (owner: Gaeun)

Qwen 3 runs through [Ollama](https://ollama.com) on Gaeun's Windows desktop (the only team GPU).
Output files use the shared result schema, so `tools/merge_check.py` and `tools/metrics.py` read them
like any API model's results.

## One-time setup (~45 min, mostly downloads)
1. Install Ollama for Windows: https://ollama.com/download (it then runs in the system tray).
2. Install Python 3.10+ if missing: https://www.python.org/downloads/ (tick "Add python.exe to PATH").
   No extra packages are needed.
3. Install the frozen model with `ollama pull qwen3:14b` on the 16 GB RTX 5060 Ti.
   The root `config.yaml` pins the full model digest; the runner checks the installed
   model before inference. A different model size requires a new protocol version.
4. Double-click `qwen\check_setup.bat`. It should end with `[OK] Setup works.`

## Before each long run
- Pause Windows Update: Settings → Windows Update → Pause updates. (The .bat files keep the PC
  awake while plugged in and set sleep back to 30 minutes when they finish.)
- Run `python -m experiments.protocol` to verify the shared config, prompts,
  examples and runner. `qwen/qwen_config.json` is a compatibility pointer to the
  root `config.yaml`. See [the protocol instructions](../prompts/README.md).

## Runs
| When | Double-click | Calls (with default settings) |
|---|---|---|
| Dev pilot | `qwen\run_qwen_dev.bat` | 379 scoring pairs: zero-shot 4,548 + few-shot 4,548 + reasoning 3,032 |
| Test phase | `qwen\run_qwen_test.bat` | 1,543 Upworthy + 500 Reddit pairs, same per-pair counts |

The window prints progress and an ETA. Completed valid JSONL calls are skipped when
the same protocol and input are resumed. A corrupt line stops with a repair message.
Greedy (temperature 0) calls for
every pair are done first, so a partial run already has the main predictions.

Speed tip: extra arguments pass through, e.g. `run_qwen_dev.bat --methods zero_shot` to run one
method, or `--workers 4` if Ollama is set to serve parallel requests (`OLLAMA_NUM_PARALLEL`).

## Settings
- Greedy call: temperature 0, seed 42. Repeats: temperature 0.7, seeds 43, 44, ... (5 repeats; 3 for reasoning).
- `zero_shot` and `few_shot` run with Qwen thinking off; `reasoning` runs with thinking on.
- Each pair runs in both orders (AB and BA). `choice` is mapped back to the pair file's A/B label.
- Answers are parsed from JSON `{"choice": 1|2, "confidence": 50-100}`; unparseable answers are
  saved with `choice: null` and counted by `merge_check.py`.
- Missing or invalid confidence is saved as null. No confidence is invented for examples.
- The dev pool excludes the four fixed example rows and three near-test duplicates;
  all methods use the same pool and derived shards. Use `data/prompt_dev.csv`
  and `data/shards/prompt_dev_shards.csv` when merging dev results.
- Prompt, protocol and model hashes are logged in every result. Results from old
  placeholder prompts cannot be resumed into the frozen protocol.
- `--dry-run` checks and renders prompts without contacting Ollama. The live
  dev pilot has not been run by the prompt-preparation work.

## Troubleshooting
- "Ollama not responding": open the Ollama app from the Start menu; the runner keeps retrying for 15 min.
- Many parse failures: check `raw_output` in the results file; usually the prompt needs the JSON line.
- Very slow: use the smaller model, or run `--methods zero_shot few_shot` first and reasoning later.
