@echo off
REM Navigate to the project directory
cd /d "%~dp0"

REM Set the relative path to the virtual environment
set ENV_PATH=HDML-env\Scripts\activate.bat

REM Activate the virtual environment
call %ENV_PATH%

REM Start Jupyter Notebook
jupyter notebook
