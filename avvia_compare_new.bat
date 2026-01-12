@echo off
cd /d "%~dp0"
set "DEFAULT_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%DEFAULT_PY%" (
    set "DEFAULT_PY=python"
)
"%DEFAULT_PY%" "%~dp0compare_new.py"
pause
