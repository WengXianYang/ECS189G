@echo off
REM Checks GPU, Ollama, Python and the Qwen model, then makes one test call.
cd /d "%~dp0.."
set PY=python
where python >nul 2>nul || set PY=py

echo === GPU ===
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if errorlevel 1 echo [!] nvidia-smi not found. Update the NVIDIA driver.

echo.
echo === Python ===
%PY% --version
if errorlevel 1 echo [!] Python not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH".

echo.
echo === Ollama ===
ollama --version
if errorlevel 1 echo [!] Ollama not found. Install it from https://ollama.com/download
ollama list

echo.
echo === Test call ===
%PY% qwen\run_qwen.py --smoke
if errorlevel 1 (echo [!] Test call failed. See qwen\README_qwen.md) else (echo [OK] Setup works.)
pause
