@echo off
setlocal enabledelayedexpansion

REM Change to the directory of this script
pushd "%~dp0"

set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR%
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :end

python -m pip install --upgrade pip
if exist requirements.txt (
    python -m pip install -r requirements.txt
)

python app_main.py

:end
popd
endlocal
