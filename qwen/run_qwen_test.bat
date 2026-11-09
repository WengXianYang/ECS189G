@echo off
REM Test phase: Qwen 3 on ALL test pairs (Upworthy, then Reddit), all methods.
REM Takes several sessions. Safe to close and re-run: finished calls are skipped.
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul || set PY=py

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0

%PY% qwen\run_qwen.py --pairs xian_data_prep\outputs\upworthy_test.csv --shards data\shards\upworthy_test_shards.csv --out-dir results\test\qwen %*
%PY% qwen\run_qwen.py --pairs reddit\reddit_pairs.csv --shards data\shards\reddit_shards.csv --out-dir results\reddit\qwen %*

powercfg /change standby-timeout-ac 30
echo Finished. Results are in results\test\qwen and results\reddit\qwen
pause
