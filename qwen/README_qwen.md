# Qwen 3 local runs (owner: Gaeun)

Qwen 3 runs through [Ollama](https://ollama.com) on Gaeun's Windows desktop (the only team GPU).
Output files use the shared result schema, so `tools/merge_check.py` and `tools/metrics.py` read them
like any API model's results.

## One-time setup (~45 min, mostly downloads)
1. Install Ollama for Windows: https://ollama.com/download (it then runs in the system tray).
2. Install Python 3.10+ if missing: https://www.python.org/downloads/ (tick "Add python.exe to PATH").
   No extra packages are needed.
3. Pick the model size from the GPU memory that `nvidia-smi` shows, then download it in a terminal:

   | GPU memory | Model | Command |
   |---|---|---|
   | 8 GB | `qwen3:8b` | `ollama pull qwen3:8b` |
   | 12–16 GB | `qwen3:14b` | `ollama pull qwen3:14b` |

   Write the tag you picked into `"model"` in `qwen/qwen_config.json` (default `qwen3:8b`). The team
   config must record the same tag.
4. Double-click `qwen\check_setup.bat`. It should end with `[OK] Setup works.`

## Before each long run
- Pause Windows Update: Settings → Windows Update → Pause updates. (The .bat files keep the PC
  awake while plugged in and set sleep back to 30 minutes when they finish.)
- Replace the placeholder prompts in `qwen/qwen_config.json` with the frozen team prompts and
  few-shot examples (after the config freeze).

## Runs
| When | Double-click | Calls (with default settings) |
|---|---|---|
| Dev pilot | `qwen\run_qwen_dev.bat` | 386 pairs: zero-shot 4,632 + few-shot 4,632 + reasoning 3,088 |
| Test phase | `qwen\run_qwen_test.bat` | 1,543 Upworthy + 500 Reddit pairs, same per-pair counts |

The window prints progress and an ETA. Closing it, a crash, or a reboot loses at most the calls in
flight: run the same .bat again and it continues where it stopped. Greedy (temperature 0) calls for
every pair are done first, so a partial run already has the main predictions.

Speed tip: extra arguments pass through, e.g. `run_qwen_dev.bat --methods zero_shot` to run one
method, or `--workers 4` if Ollama is set to serve parallel requests (`OLLAMA_NUM_PARALLEL`).

## Settings
- Greedy call: temperature 0, seed 42. Repeats: temperature 0.7, seeds 43, 44, ... (5 repeats; 3 for reasoning).
- `zero_shot` and `few_shot` run with Qwen thinking off; `reasoning` runs with thinking on.
- Each pair runs in both orders (AB and BA). `choice` is mapped back to the pair file's A/B label.
- Answers are parsed from JSON `{"choice": 1|2, "confidence": 50-100}`; unparseable answers are
  saved with `choice: null` and counted by `merge_check.py`.

## Troubleshooting
- "Ollama not responding": open the Ollama app from the Start menu; the runner keeps retrying for 15 min.
- Many parse failures: check `raw_output` in the results file; usually the prompt needs the JSON line.
- Very slow: use the smaller model, or run `--methods zero_shot few_shot` first and reasoning later.
