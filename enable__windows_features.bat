@echo off
cd /D "%~dp0"

:: Check if the script is running as administrator
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Requesting administrative privileges...
    powershell start-process '%0' -verb runas
    exit /B
)

:: Function to check and enable a Windows feature
:CheckAndEnableFeature
setlocal
set "featureName=%~1"
echo Checking feature: %featureName%
Dism /online /Get-FeatureInfo /FeatureName:%featureName% | findstr /C:"State : Enabled" >nul
if %ERRORLEVEL% EQU 0 (
    echo %featureName% is already enabled.
) else (
    echo Enabling %featureName%...
    Dism /online /Enable-Feature /FeatureName:%featureName% /All
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to enable %featureName%.
    ) else (
        echo %featureName% enabled successfully.
    )
)
endlocal
exit /B

:: Check and enable Hyper-V features
echo Checking and enabling Hyper-V features...
call :CheckAndEnableFeature Microsoft-Hyper-V-Tools-All
pause
call :CheckAndEnableFeature Microsoft-Hyper-V-Management-PowerShell
pause
call :CheckAndEnableFeature Microsoft-Hyper-V-Hypervisor
pause
call :CheckAndEnableFeature Microsoft-Hyper-V-Services
pause
call :CheckAndEnableFeature Microsoft-Hyper-V-Management-Clients
pause

:: Check and enable Virtual Machine Platform
echo Checking and enabling Virtual Machine Platform...
call :CheckAndEnableFeature HypervisorPlatform
pause

:: Check and enable Windows Subsystem for Linux
echo Checking and enabling Windows Subsystem for Linux...
call :CheckAndEnableFeature Microsoft-Windows-Subsystem-Linux
pause

echo All required features are enabled.

:: Done
echo Done!
pause