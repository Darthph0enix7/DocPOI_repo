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
            param_manager.set_param("selected_gpu", selected_gpu)
            print(f"Using GPU: {system_info['gpu_info'][selected_gpu]['name']}")
            handle_selected_gpu(selected_gpu)  # Call the function with the GPU index
        else:
            print("Invalid choice. Using all GPUs.")
            param_manager.set_param("selected_gpu", "all")
            handle_selected_gpu("all")  # Call the function with "all"
    elif system_info["gpu_count"] == 1:
        print(f"Using GPU: {system_info['gpu_info'][0]['name']}")
        param_manager.set_param("selected_gpu", 0)
        handle_selected_gpu(0)  # Call the function with the GPU index
    else:
        print("No GPU detected. Running in CPU mode.")
        param_manager.set_param("selected_gpu", "cpu")

def handle_selected_gpu(gpu_index):
    if platform.system() == "Windows":
        # Set CUDA_VISIBLE_DEVICES in the global system environment on Windows
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index) if gpu_index != "all" else ",".join(map(str, range(psutil.cpu_count(logical=True))))
        print(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']} on Windows")
    elif platform.system() == "Linux":
        # Edit the /etc/systemd/system/ollama.service file on Linux
        service_file = "/etc/systemd/system/ollama.service"
        
        # Read the current content of the service file
        with open(service_file, "r") as file:
            lines = file.readlines()

        # Prepare the new content
        new_lines = []
        service_section_found = False
        for line in lines:
            if line.strip() == "[Service]":
                service_section_found = True
            if service_section_found and line.startswith("Environment=\"CUDA_VISIBLE_DEVICES="):
                continue  # Skip the existing CUDA_VISIBLE_DEVICES line
            new_lines.append(line)
            if service_section_found and line.strip() == "":
                # Add the new CUDA_VISIBLE_DEVICES line after the [Service] section
                cuda_visible_devices = str(gpu_index) if gpu_index != "all" else ",".join(map(str, range(psutil.cpu_count(logical=True))))
                new_lines.append(f'Environment="CUDA_VISIBLE_DEVICES={cuda_visible_devices}"\n')
                service_section_found = False

        # Write the new content to a temporary file
        temp_file = "/tmp/ollama.service"
        with open(temp_file, "w") as file:
            file.writelines(new_lines)

        # Move the temporary file to the service file location with sudo
        run_cmd(f"sudo mv {temp_file} {service_file}")

        # Reload the systemd daemon and restart the service
        run_cmd("sudo systemctl daemon-reload")
        run_cmd("sudo systemctl restart ollama.service")
        print(f"Set CUDA_VISIBLE_DEVICES to {cuda_visible_devices} in /etc/systemd/system/ollama.service on Linux")
    else:
        print("Unsupported operating system. Exiting...")
        sys.exit()

def install_poppler():
    if platform.system() == 'Windows':
        install_dir = os.getcwd()
        poppler_path = os.path.join(install_dir, 'poppler-24.07.0')
        poppler_download_url = 'https://github.com/oschwartz10612/poppler-windows/releases/download/v24.07.0-0/Release-24.07.0-0.zip'
        poppler_zip_path = os.path.join(install_dir, 'poppler.zip')

        if not os.path.exists(poppler_path):
            print(f"Downloading Poppler from {poppler_download_url} to {poppler_zip_path}")
            run_cmd(f"curl -L -o {poppler_zip_path} {poppler_download_url}")
            print(f"Unzipping Poppler to {install_dir}")
            run_cmd(f"tar -xf {poppler_zip_path} -C {install_dir}")
        else:
            print(f"Poppler is already unzipped at {poppler_path}.")
    elif platform.system() == 'Linux':
        poppler_installed = run_cmd("command -v pdftocairo", capture_output=True).returncode == 0

        if not poppler_installed:
            print("Installing Poppler utilities...")
            run_cmd("sudo apt-get install -y poppler-utils")
        else:
            print("Poppler utilities are already installed.")
    else:
        print("Unsupported operating system for Poppler installation. Exiting...")
        sys.exit()

def install_tesseract():
    install_dir = os.path.join(os.getcwd(), 'installer_files')
    os.makedirs(install_dir, exist_ok=True)

    if platform.system() == 'Windows':
        tesseract_path = os.path.join(os.environ['ProgramFiles'], 'Tesseract-OCR', 'tesseract.exe')
        tesseract_download_url = 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.4.0.20240606.exe'
        tesseract_installer_path = os.path.join(install_dir, 'tesseract_installer.exe')

        if not os.path.exists(tesseract_path):
            print(f"Downloading Tesseract from {tesseract_download_url} to {tesseract_installer_path}")
            run_cmd(f"curl -L -o {tesseract_installer_path} {tesseract_download_url}")
            print("Installing Tesseract silently")
            run_cmd(f"{tesseract_installer_path} /S")
        else:
            print(f"Tesseract is already installed at {tesseract_path}.")
    elif platform.system() == 'Linux':
        tesseract_installed = run_cmd("command -v tesseract", capture_output=True).returncode == 0

        if not tesseract_installed:
            print("Installing Tesseract OCR...")
            run_cmd("sudo apt-get install -y tesseract-ocr")
        else:
            print("Tesseract is already installed.")

    # Download tessdata
    tessdata_repo_url = 'https://github.com/Darthph0enix7/Tesseract_Tessdata_current.git'
    tessdata_path = os.path.join(install_dir, 'tessdata')

    if not os.path.exists(tessdata_path):
        print(f"Cloning tessdata repository from {tessdata_repo_url} to {tessdata_path}")
        run_cmd(f"git clone {tessdata_repo_url} {tessdata_path}")
        print("Tessdata successfully cloned.")
    else:
        print(f"Tessdata is already downloaded at {tessdata_path}.")

def install_requirements():
    requirements_path = os.path.join(repo_dir, 'requirements.txt')
    if os.path.exists(requirements_path):
        print("Installing requirements from requirements.txt...")
        run_cmd(f"pip install -r {requirements_path}")
    else:
        print("requirements.txt not found. Skipping installation of requirements.")
        
def initial_setup():
    # Select your GPU or, choose to run in CPU mode
    print()
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
        
        # Gather system information after installing PyTorch
        system_info = gather_system_info()

        # Check if the system meets the recommended specs
        check_recommended_specs(system_info)

        # Allow the user to select a GPU if multiple are available
        select_gpu(system_info)
    elif gpuchoice == "b":
        print("AMD GPUs are not supported yet. Try CPU installation. Exiting...")
        sys.exit()
    elif gpuchoice == "c" or gpuchoice == "d":
        run_cmd("conda install -y -k pytorch torchvision torchaudio cpuonly ninja git curl -c pytorch")
        
        # Gather system information after installing PyTorch
        system_info = gather_system_info()

        # Check if the system meets the recommended specs
        check_recommended_specs(system_info)
    else:
        print("Invalid choice. Exiting...")
        sys.exit()

    # Install requirements
    install_requirements()

    # Install Poppler
    install_poppler()

    # Install Tesseract
    install_tesseract()

    #install the requirements.txt

    # Mark setup as completed
    with open(setup_flag_file, "w") as f:
        f.write("Setup completed")

def run_main():
    os.chdir(repo_dir)
    run_cmd("python main.py")  # put your flags here!

if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    # Check if setup has already been completed
    if not os.path.exists(setup_flag_file):
        initial_setup()

    # Run the model
    run_main()