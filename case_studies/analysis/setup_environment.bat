@echo off
REM TI-Toolbox Research Environment Setup Script for Windows
REM This script creates a virtual environment and installs all dependencies

echo ==========================================
echo TI-Toolbox Research Environment Setup
echo ==========================================

REM Check if Python 3 is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Verify installation
echo Verifying installation...
python -c "import pandas, numpy, scipy, statsmodels, matplotlib, seaborn; print('All core packages installed successfully!')"

echo ==========================================
echo Environment setup completed!
echo ==========================================
echo.
echo To activate the virtual environment:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate the virtual environment:
echo   deactivate
echo.
echo To run the analysis scripts:
echo   python scripts\analysis\run_all_analyses.py --help
echo.
echo ==========================================
pause 