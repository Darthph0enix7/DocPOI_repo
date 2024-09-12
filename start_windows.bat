@echo off

cd /D "%~dp0"

set "currentDir=%CD%"
echo %currentDir% | findstr " " >nul
if "%ERRORLEVEL%" == "0" (
    echo This script relies on Miniconda which cannot be silently installed under a path with spaces.
    goto end
)

set PATH=%PATH%;%SystemRoot%\system32

@rem config
set DISTUTILS_USE_SDK=1

set INSTALL_DIR=%cd%\installer_files
set CONDA_ROOT_PREFIX=%cd%\installer_files\conda
set INSTALL_ENV_DIR=%cd%\installer_files\env
set TTS_REPO_DIR=%cd%\DocPOI_repo\XTTS-v2
set MINICONDA_DOWNLOAD_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Windows-x86_64.exe
set TESSERACT_PATH=%ProgramFiles%\Tesseract-OCR\tesseract.exe
set OLLAMA_PATH=%LocalAppData%\Programs\Ollama\ollama app.exe
set POPPLER_PATH=%INSTALL_DIR%\poppler-24.07.0
set DOCKER_PATH=%ProgramFiles%\Docker\Docker\Docker Desktop.exe
set TESSERACT_DOWNLOAD_URL=https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.4.0.20240606.exe
set OLLAMA_DOWNLOAD_URL=https://ollama.com/download/OllamaSetup.exe
set POPPLER_DOWNLOAD_URL=https://github.com/oschwartz10612/poppler-windows/releases/download/v24.07.0-0/Release-24.07.0-0.zip
set TESSDATA_REPO_URL=https://github.com/tesseract-ocr/tessdata.git
set TESSCONFIGS_REPO_URL=https://github.com/tesseract-ocr/tessconfigs.git
set conda_exists=F
set VS_BUILD_TOOLS_URL=https://aka.ms/vs/17/release/vs_BuildTools.exe
set "MSBUILD_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin"

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
    call python -m pip install requests || ( echo. && echo Failed to install requests. && goto end )
)

@rem Check for and install Tesseract if not installed
if not exist "%TESSERACT_PATH%" (
    echo Downloading Tesseract from %TESSERACT_DOWNLOAD_URL% to %INSTALL_DIR%\tesseract_installer.exe
    curl -L -o "%INSTALL_DIR%\tesseract_installer.exe" %TESSERACT_DOWNLOAD_URL% || ( echo. && echo Tesseract failed to download. && goto end )

    echo Installing Tesseract silently
    pushd "%INSTALL_DIR%"
    tesseract_installer.exe /S
    popd
) else (
    echo Tesseract is already installed at %TESSERACT_PATH%.
)

@rem Check for and download tessdata and tessconfigs if not already downloaded
if not exist "%INSTALL_DIR%\tessdata" (
    echo Downloading tessdata from %TESSDATA_REPO_URL% to %INSTALL_DIR%\tessdata
    git clone %TESSDATA_REPO_URL% "%INSTALL_DIR%\tessdata" || ( echo. && echo Tessdata failed to download. && goto end )

    echo Downloading tessconfigs from %TESSCONFIGS_REPO_URL% to %INSTALL_DIR%\tessdata\tessconfigs
    git clone %TESSCONFIGS_REPO_URL% "%INSTALL_DIR%\tessdata\tessconfigs" || ( echo. && echo Tessconfigs failed to download. && goto end )

    @rem Remove the configs file in tessdata
    if exist "%INSTALL_DIR%\tessdata\configs" (
        del "%INSTALL_DIR%\tessdata\configs" || ( echo. && echo Failed to delete configs file. && goto end )
    )

    @rem Move the configs folder from tessconfigs to the root tessdata folder
    if exist "%INSTALL_DIR%\tessdata\tessconfigs\configs" (
        move "%INSTALL_DIR%\tessdata\tessconfigs\configs" "%INSTALL_DIR%\tessdata" || ( echo. && echo Failed to move configs folder. && goto end )
    )
) else (
    echo Tessdata and tessconfigs are already downloaded at %INSTALL_DIR%\tessdata.
)
@rem Check for and install Ollama if not installed
if not exist "%OLLAMA_PATH%" (
    echo Downloading Ollama from %OLLAMA_DOWNLOAD_URL% to %INSTALL_DIR%\ollama_installer.exe
    curl -L -o "%INSTALL_DIR%\ollama_installer.exe" %OLLAMA_DOWNLOAD_URL% || ( echo. && echo Ollama failed to download. && goto end )

    echo Installing Ollama
    "%INSTALL_DIR%\ollama_installer.exe" /S

    @rem Pull the model in a new terminal and continue with the rest of the installation
    start "" cmd /c "ollama pull llama3.1:8b"
) else (
    echo Ollama is already installed at %OLLAMA_PATH%.
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

@rem Check if MSBuild exists in the specified path
if exist "%MSBUILD_PATH%\MSBuild.exe" (
    echo MSBuild found in the specified path.
) else (
    echo MSBuild not found in the specified path, checking Visual Studio installation...

    @rem Check for Visual Studio Build Tools installation using vswhere
    vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 >nul 2>&1

    if "%ERRORLEVEL%" NEQ "0" (
        echo Visual Studio c++ Build Tools not found. Downloading and installing...

        curl -L -o "%INSTALL_DIR%\vs_buildtools.exe" %VS_BUILD_TOOLS_URL% || (
            echo.
            echo Visual Studio Build Tools failed to download.
            goto end
        )

        echo Installing Visual Studio Build Tools silently. This process may take a while.
        "%INSTALL_DIR%\vs_buildtools.exe" --quiet --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.ManagedDesktopBuildTools --add Microsoft.VisualStudio.Workload.WebBuildTools --add Microsoft.VisualStudio.Workload.NetCoreBuildTools --add Microsoft.VisualStudio.Component.Windows10SDK.18362 --add Microsoft.Net.Component.4.7.TargetingPack --includeRecommended --wait || (
            echo.
            echo Visual Studio Build Tools installation failed.
            goto end
        )
    ) else (
        echo Visual Studio c++ Build Tools are already installed.
    )
)

@rem Check for and install Docker if not installed
if not exist "%DOCKER_PATH%" (
    echo Docker is not installed. Proceeding with Docker installation...
    
    @rem Download Docker installer using Python
    call python webui.py --download-docker

    @rem Run the PowerShell script to install Docker
    powershell -ExecutionPolicy Bypass -File "%cd%\install_docker.ps1" -dockerInstallerPath "%INSTALL_DIR%\docker-installer.exe"
) else (
    echo Docker is already installed at %DOCKER_PATH%.
)

@rem run the Docker Installation
call python webui.py --setup-elasticsearch

@rem run the Docker Installation
call python webui.py --run-ollama

@rem run the Python script
call python webui.py %*

echo.
echo Done!

:end
pause
