import subprocess
import os
import sys

# Set CUDA environment for the subprocess
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"

# Ensure compatibility for Windows & Linux
creation_flags = 0
if sys.platform == "win32":
    creation_flags = subprocess.CREATE_NO_WINDOW  # Prevents opening a new terminal window

# Run the process in the background **without logs**
subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,  # No logs
    stderr=subprocess.DEVNULL,  # No error messages
    stdin=subprocess.DEVNULL,   # Prevents user input
    env=env,  # Pass CUDA environment
    shell=True if sys.platform == "win32" else False,  # Needed for Windows
    creationflags=creation_flags if sys.platform == "win32" else 0,
    start_new_session=True if sys.platform != "win32" else False  # Detach from parent
)

# Python exits, but the server keeps running
print("Server started in the background.")
