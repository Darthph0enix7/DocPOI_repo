@echo off
setlocal

set "src_dir=%~1"
set "dest_dir=%~2"

:: Check if the script is running with elevated privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c elevate_and_run.bat \"%src_dir%\" \"%dest_dir%\"' -Verb RunAs"
    exit /b
)

:: Run the Python script with elevated privileges
"%~dp0installer_files\env\python.exe" "%~dp0copy_tessdata.py" "%src_dir%" "%dest_dir%"
pause