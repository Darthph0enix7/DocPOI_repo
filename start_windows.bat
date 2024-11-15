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
set OLLAMA_PATH=%LocalAppData%\Programs\Ollama\ollama.exe
set OLLAMA_DOWNLOAD_URL=https://ollama.com/download/OllamaSetup.exe
set POPPLER_DOWNLOAD_URL=https://github.com/oschwartz10612/poppler-windows/releases/download/v24.07.0-0/Release-24.07.0-0.zip
set POPPLER_PATH=%INSTALL_DIR%\poppler-24.07.0
set TESSDATA_REPO_URL=https://github.com/Darthph0enix7/Tesseract_Tessdata_current.git

set conda_exists=F

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

@rem Check for and install Ollama if not installed
if not exist "%OLLAMA_PATH%" (
    echo Downloading Ollama from %OLLAMA_DOWNLOAD_URL% to %INSTALL_DIR%\ollama_installer.exe
    curl -L -o "%INSTALL_DIR%\ollama_installer.exe" %OLLAMA_DOWNLOAD_URL% || ( echo. && echo Ollama failed to download. && goto end )

    echo Installing Ollama
    "%INSTALL_DIR%\ollama_installer.exe" /S
) else (
    echo Ollama is already installed at %OLLAMA_PATH%.
)

@rem Check for and download tessdata if not already downloaded
if not exist "%INSTALL_DIR%\tessdata" (
    echo Cloning tessdata repository from %TESSDATA_REPO_URL% to %INSTALL_DIR%\tessdata
    git clone %TESSDATA_REPO_URL% "%INSTALL_DIR%\tessdata" || ( echo. && echo Tessdata repository failed to clone. && goto end )

    echo Tessdata successfully cloned.
) else (
    echo Tessdata is already downloaded at %INSTALL_DIR%.
)

@rem Check for and unzip Poppler if not already unzipped
if not exist "%POPPLER_PATH%" (
    echo Downloading Poppler from %POPPLER_DOWNLOAD_URL% to %INSTALL_DIR%\poppler.zip
    curl -L -o "%INSTALL_DIR%\poppler.zip" %POPPLER_DOWNLOAD_URL% || ( echo. && echo Poppler failed to download. && goto end )

    echo Unzipping Poppler to %INSTALL_DIR%
    tar -xf "%INSTALL_DIR%\poppler.zip" -C "%INSTALL_DIR%"
) else (
    echo Poppler is already unzipped at %POPPLER_PATH%.
)

@rem run the Python script
call python setup.py %*

echo.
echo Done!

:end
pause