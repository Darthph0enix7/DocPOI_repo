import os
import subprocess
import sys
import platform
import psutil
from components.param_manager import ParamManager

param_manager = ParamManager()

script_dir = os.getcwd()
repo_dir = os.path.join(script_dir)
setup_flag_file = os.path.join(repo_dir, "setup_completed.flag")

RECOMMENDED_SPECS = {
    "gpu_vram": 10 * 1024**3,  # 10 GB in bytes
    "cpu_threads": 4,
    "ram": 16 * 1024**3  # 16 GB in bytes
}

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

def gather_system_info():
    # Gather system information
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram": psutil.virtual_memory().total,
        "gpu_count": 0,
        "gpu_info": []
    }

    # Check for GPU information
    try:
        import torch
        if torch.cuda.is_available():
            system_info["gpu_count"] = torch.cuda.device_count()
            for i in range(system_info["gpu_count"]):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_vram = torch.cuda.get_device_properties(i).total_memory
                system_info["gpu_info"].append({"name": gpu_name, "vram": gpu_vram})
    except ImportError:
        pass

    # Set parameters using param_manager
    for key, value in system_info.items():
        param_manager.set_param(key, value)

    return system_info

def check_recommended_specs(system_info):
    # Check if the system meets the recommended specs
    meets_specs = True

    if system_info["cpu_count"] < RECOMMENDED_SPECS["cpu_threads"]:
        print(f"Warning: Your CPU has {system_info['cpu_count']} threads. Recommended: {RECOMMENDED_SPECS['cpu_threads']} threads.")
        meets_specs = False

    if system_info["ram"] < RECOMMENDED_SPECS["ram"]:
        print(f"Warning: Your system has {system_info['ram'] / 1024**3:.2f} GB of RAM. Recommended: {RECOMMENDED_SPECS['ram'] / 1024**3:.2f} GB.")
        meets_specs = False

    if system_info["gpu_count"] > 0:
        for i, gpu in enumerate(system_info["gpu_info"]):
            if gpu["vram"] < RECOMMENDED_SPECS["gpu_vram"]:
                print(f"Warning: Your GPU {gpu['name']} has {gpu['vram'] / 1024**3:.2f} GB of VRAM. Recommended: {RECOMMENDED_SPECS['gpu_vram'] / 1024**3:.2f} GB.")
                meets_specs = False
    else:
        print("Warning: No GPU detected. Recommended: At least one GPU with more than 10 GB of VRAM.")
        meets_specs = False

    return meets_specs

def select_gpu(system_info):
    if system_info["gpu_count"] > 1:
        print("Multiple GPUs detected:")
        for i, gpu in enumerate(system_info["gpu_info"]):
            print(f"{i}: {gpu['name']} with {gpu['vram'] / 1024**3:.2f} GB of VRAM")

        gpu_choice = input("Enter the number of the GPU you want to use (or press Enter to use all GPUs): ")
        if gpu_choice.isdigit() and 0 <= int(gpu_choice) < system_info["gpu_count"]:
            selected_gpu = int(gpu_choice)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            print(f"Using GPU: {system_info['gpu_info'][selected_gpu]['name']}")
        else:
            print("Invalid choice. Using all GPUs.")
    elif system_info["gpu_count"] == 1:
        print(f"Using GPU: {system_info['gpu_info'][0]['name']}")
    else:
        print("No GPU detected. Running in CPU mode.")

def initial_setup():
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
        run_cmd("conda install -y -k pytorch torchvision torchaudio pytorch-cuda=12.1 ninja git curl -c pytorch -c nvidia")
    elif gpuchoice == "b":
        print("AMD GPUs are not supported yet. Try CPU installation. Exiting...")
        sys.exit()
    elif gpuchoice == "c" or gpuchoice == "d":
        run_cmd("conda install -y -k pytorch torchvision torchaudio cpuonly ninja git curl -c pytorch")
    else:
        print("Invalid choice. Exiting...")
        sys.exit()

    # Gather system information after installing PyTorch
    system_info = gather_system_info()

    # Check if the system meets the recommended specs
    check_recommended_specs(system_info)

    # Allow the user to select a GPU if multiple are available
    select_gpu(system_info)

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
        initial_setup()

    # Run the model
    run_model()
