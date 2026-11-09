@echo off
REM Dev pilot: scoring pool excludes demonstrations and near-test duplicates.
REM Safe to close and re-run: finished calls are skipped.
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul || set PY=py

REM keep the PC awake while plugged in (restored to 30 min at the end)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

%PY% qwen\run_qwen.py --pairs data\prompt_dev.csv --shards data\shards\prompt_dev_shards.csv --out-dir results\dev\qwen %*

powercfg /change standby-timeout-ac 30
echo Finished. Results are in results\dev\qwen
pause
