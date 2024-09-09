#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Run the main.py script
python "$SCRIPT_DIR/main.py"