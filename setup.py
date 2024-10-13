import os
import subprocess
import sys
import argparse

script_dir = os.getcwd()
repo_dir = os.path.join(script_dir)
setup_flag_file = os.path.join(repo_dir, "setup_completed.flag")


def run_cmd(cmd, capture_output=False, env=None):
    # Run shell commands
    return subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_not_exist = run_cmd("conda", capture_output=True).returncode
    if conda_not_exist:
        print("Conda is not installed. Exiting...")
        sys.exit()

    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit()


def install_dependencies():
    # Select your GPU or, choose to run in CPU mode
    print("What is your GPU")
    print()
    print("A) NVIDIA")
    print("B) AMD")
    print("C) Apple M Series")
    print("D) None (I want to run in CPU mode)")
    print()
    gpuchoice = input("Input> ").lower()

    # Install the version of PyTorch needed
    if gpuchoice == "a":
        run_cmd("conda install -y -k nvidia/label/cuda-12.1.0::cuda-toolkit")
        run_cmd("conda install -y -k pytorch torchvision torchaudio pytorch-cuda=12.1 ninja git -c pytorch -c nvidia")
    elif gpuchoice == "b":
        print("AMD GPUs are not supported. Exiting...")
        sys.exit()
    elif gpuchoice == "c" or gpuchoice == "d":
        run_cmd("conda install -y -k pytorch torchvision torchaudio cpuonly git -c pytorch")
    else:
        print("Invalid choice. Exiting...")
        sys.exit()

    # Mark setup as completed
    with open(setup_flag_file, "w") as f:
        f.write("Setup completed")


def run_model():
    os.chdir(repo_dir)
    run_cmd("python main.py")  # put your flags here!


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    # Check if setup has already been completed
    if not os.path.exists(setup_flag_file):
        install_dependencies()

    # Run the model
    run_model()
