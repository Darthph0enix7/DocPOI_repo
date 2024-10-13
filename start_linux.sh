#!/usr/bin/env bash

# Navigate to the script's directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Check if the script is located in a directory with spaces
if [[ "$(pwd)" =~ " " ]]; then 
    echo "This script relies on Miniconda which cannot be silently installed under a path with spaces."
    exit 1
fi

# Deactivate existing conda environments to avoid conflicts
{ conda deactivate && conda deactivate && conda deactivate; } 2>/dev/null

# Determine the system architecture
OS_ARCH=$(uname -m)
case "${OS_ARCH}" in
    x86_64*) OS_ARCH="x86_64";;
    arm64*|aarch64*) OS_ARCH="aarch64";;
    *) echo "Unknown system architecture: $OS_ARCH! This script runs only on x86_64 or arm64." && exit 1;;
esac

# Configuration
INSTALL_DIR="$(pwd)/installer_files"
CONDA_ROOT_PREFIX="$INSTALL_DIR/conda"
INSTALL_ENV_DIR="$INSTALL_DIR/env"
MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-${OS_ARCH}.sh"
conda_exists="F"

# Check if Conda needs to be installed
if "$CONDA_ROOT_PREFIX/bin/conda" --version &>/dev/null; then 
    conda_exists="T"
fi

# Install Conda if necessary
if [ "$conda_exists" == "F" ]; then
    echo "Downloading Miniconda from $MINICONDA_DOWNLOAD_URL to $INSTALL_DIR/miniconda_installer.sh"
    
    mkdir -p "$INSTALL_DIR"
    curl -L "$MINICONDA_DOWNLOAD_URL" > "$INSTALL_DIR/miniconda_installer.sh"
    
    chmod u+x "$INSTALL_DIR/miniconda_installer.sh"
    bash "$INSTALL_DIR/miniconda_installer.sh" -b -p "$CONDA_ROOT_PREFIX"
    
    echo "Miniconda version:"
    "$CONDA_ROOT_PREFIX/bin/conda" --version
    
    # Optionally, remove the Miniconda installer
    rm "$INSTALL_DIR/miniconda_installer.sh"
fi

# Create the installer environment if it doesn't exist
if [ ! -d "$INSTALL_ENV_DIR" ]; then
    "$CONDA_ROOT_PREFIX/bin/conda" create -y -k --prefix "$INSTALL_ENV_DIR" python=3.11.5
fi

# Check if Conda environment was actually created
if [ ! -e "$INSTALL_ENV_DIR/bin/python" ]; then
    echo "Conda environment is empty."
    exit 1
fi

# Environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

# Activate the installer environment
source "$CONDA_ROOT_PREFIX/etc/profile.d/conda.sh"
conda activate "$INSTALL_ENV_DIR"

# Ensure necessary Python modules are installed
python -c "import requests" 2>/dev/null || python -m pip install requests

# Run the main Python script
python setup.py "$@"

# End of the script
echo "Done!"