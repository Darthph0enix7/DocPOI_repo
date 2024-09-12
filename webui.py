import argparse
import os
import subprocess
import sys
import shutil
import time
import requests

script_dir = os.getcwd()
repo_dir = os.path.join(script_dir)
remote_url = "https://github.com/Darthph0enix7/DocPOI_repo.git"
tts_repo_dir = os.path.join(repo_dir, "XTTS-v2")


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
        run_cmd(
            "conda install -y -k nvidia/label/cuda-12.1.0::cuda-toolkit"
        )
        run_cmd(
            "conda install -y -k pytorch torchvision torchaudio pytorch-cuda=12.1  ninja git -c pytorch -c nvidia"
        )
    elif gpuchoice == "b":
        print("AMD GPUs are not supported. Exiting...")
        sys.exit()
    elif gpuchoice == "c" or gpuchoice == "d":
        run_cmd(
            "conda install -y -k pytorch torchvision torchaudio cpuonly git -c pytorch"
        )
    else:
        print("Invalid choice. Exiting...")
        sys.exit()
        
    run_cmd("conda install -y -c pytorch ffmpeg")  # LGPL

    # Install the webui dependencies
    update_dependencies()

    # Install Git LFS if not installed
    if run_cmd("git lfs --version", capture_output=True).returncode != 0:
        print("Git LFS is not installed. Installing Git LFS...")
        run_cmd("git lfs install")
    
    # Clone the XTTS-v2 repository if it doesn't already exist
    if not os.path.exists(tts_repo_dir):
        print(f"Cloning the XTTS-v2 repository into {tts_repo_dir}...")
        run_cmd(f"git clone https://huggingface.co/coqui/XTTS-v2 {tts_repo_dir}")
    else:
        print("XTTS-v2 repository already exists.")


def setup_elasticsearch():
    # Use environment variable to construct the Docker Desktop path
    docker_path = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Docker", "Docker", "Docker Desktop.exe")

    if not os.path.exists(docker_path):
        print(f"Docker Desktop is not installed at the expected path: {docker_path}")
        sys.exit(1)

    # Check if Docker is running
    print("Checking if Docker is running...")
    docker_running = subprocess.run(["docker", "info"], capture_output=True).returncode == 0

    if not docker_running:
        print("Docker is not running. Starting Docker Desktop...")
        subprocess.Popen([docker_path])

        # Wait for Docker to be ready
        print("Waiting for Docker to start...")
        while not docker_running:
            time.sleep(5)  # Check every 5 seconds
            docker_running = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
        print("Docker is now running.")
    else:
        print("Docker is already running.")

    # Check if an Elasticsearch container exists
    print("Checking if an Elasticsearch container already exists...")
    result = subprocess.run(["docker", "ps", "-a", "--filter", "ancestor=docker.elastic.co/elasticsearch/elasticsearch:8.12.1", "--format", "{{.ID}}"], capture_output=True, text=True)

    container_id = result.stdout.strip()

    if container_id:
        # Check if the container is running
        print("Checking if the existing Elasticsearch container is running...")
        result = subprocess.run(["docker", "ps", "--filter", f"id={container_id}", "--format", "{{.ID}}"], capture_output=True, text=True)

        if result.stdout.strip():
            print("Elasticsearch container is already running.")
        else:
            # Start the existing container
            print("Starting the existing Elasticsearch container...")
            subprocess.run(["docker", "start", container_id])
            print("Elasticsearch container started.")
    else:
        # Run a new Elasticsearch Docker container in detached mode (background)
        print("No existing Elasticsearch container found. Running a new one in the background...")
        subprocess.Popen('docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.1', shell=True)
        print("New Elasticsearch container started in the background.")
    
    # Close the command prompt
    print("Closing command prompt...")
    sys.exit(0)

def run_ollama():
    # Run the Ollama command
    print("Starting Ollama with llama3.1:8b...")
    process = subprocess.Popen(["ollama", "run", "llama3.1:8b"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def update_conda():
    # Update conda
    run_cmd("conda update -y -n base -c defaults conda")


def update_dependencies():
    # Update the webui dependencies
    os.chdir(repo_dir)
    
    # Check if the .git directory exists
    if not os.path.isdir(".git"):
        print("Initializing new Git repository...")
        run_cmd("git init")
        run_cmd(f"git remote add origin {remote_url}")
    
    # Ensure the repository is connected to the remote
    run_cmd("git fetch origin")
    run_cmd("git checkout main")
    run_cmd("git pull origin main")
    
    # Install dependencies
    run_cmd("pip install -r requirements.txt")
    
    os.chdir(script_dir)


def download_docker_installer():
    docker_installer_url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-win-amd64"
    docker_installer_path = os.path.join(script_dir, "installer_files", "docker-installer.exe")

    # Create installer_files directory if it doesn't exist
    os.makedirs(os.path.dirname(docker_installer_path), exist_ok=True)

    # Download Docker installer
    print("Downloading Docker Desktop...")
    try:
        response = requests.get(docker_installer_url, stream=True)
        response.raise_for_status()
        with open(docker_installer_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print("Docker Desktop downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download Docker: {e}")
        sys.exit(1)

    return docker_installer_path


def run_model():
    os.chdir(repo_dir)
    run_cmd("python main.py")  # put your flags here!


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--download-docker", action="store_true", help="Download Docker Desktop installer.")
    parser.add_argument("--update", action="store_true", help="Update the web UI.")
    parser.add_argument("--setup-elasticsearch", action="store_true", help="Setup and run Elasticsearch in Docker.")
    parser.add_argument("--run-ollama", action="store_true", help="Run and terminate Ollama with llama3.1:8b.")
    args = parser.parse_args()

    if args.update:
        update_dependencies()
    elif args.download_docker:
        download_docker_installer()
    elif args.setup_elasticsearch:
        setup_elasticsearch()
    elif args.run_ollama:
        run_ollama()
    else:
        # If webui has already been installed, skip and run
        if not os.path.exists(tts_repo_dir):
            install_dependencies()
            os.chdir(script_dir)

        # Run the model with webui
        run_model()
