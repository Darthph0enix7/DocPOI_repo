@echo off

cd /D "%~dp0"

set "currentDir=%CD%"
echo %currentDir% | findstr " " >nul
if "%ERRORLEVEL%" == "0" (
    echo This script relies on Miniconda which cannot be silently installed under a path with spaces.
    goto end
)

@echo off
set PATH=%PATH%;%SystemRoot%\system32

@rem config
set DISTUTILS_USE_SDK=1

set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
@rem figure out whether conda needs to be installed
call "%CONDA_ROOT_PREFIX%\_conda.exe" --version >nul 2>&1
if "%ERRORLEVEL%" EQU "0" set conda_exists=T

@rem (if necessary) install conda into a contained environment
if "%conda_exists%" == "F" (
    echo Downloading Miniconda from %MINICONDA_DOWNLOAD_URL% to %INSTALL_DIR%\miniconda_installer.exe

    mkdir "%INSTALL_DIR%"
    curl -L -o "%INSTALL_DIR%\miniconda_installer.exe" %MINICONDA_DOWNLOAD_URL% || ( echo. && echo Miniconda failed to download. && goto end )

    echo Installing Miniconda to %CONDA_ROOT_PREFIX%
    "%INSTALL_DIR%\miniconda_installer.exe" /InstallationType=JustMe /NoShortcuts=1 /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%CONDA_ROOT_PREFIX%

    @rem test the conda binary
    echo Miniconda version:
    call "%CONDA_ROOT_PREFIX%\_conda.exe" --version || ( echo. && echo Miniconda not found. && goto end )
)

@rem create the installer env if it doesn't exist
if not exist "%INSTALL_ENV_DIR%" (
    echo Creating the conda environment...
    call "%CONDA_ROOT_PREFIX%\_conda.exe" create --no-shortcuts -y -k --prefix "%INSTALL_ENV_DIR%" python=3.11.5 || ( echo. && echo Conda environment creation failed. && goto end )
)

@rem check if conda environment was actually created
if not exist "%INSTALL_ENV_DIR%\python.exe" ( echo. && echo Conda environment is empty. && goto end )

@rem activate installer env
call "%CONDA_ROOT_PREFIX%\condabin\conda.bat" activate "%INSTALL_ENV_DIR%" || ( echo. && echo Miniconda hook not found. && goto end )

@rem Check if requests is installed; if not, install it
call python -c "import requests" 2>nul || (
    echo Installing requests module...
    call python -m pip install requests psutil || ( echo. && echo Failed to install requests. && goto end )
)

@rem run the Python script
call python setup.py %*

echo.
echo Done!

:end
pause
