param(
    [string]$dockerInstallerPath
)

if (-not (Test-Path $dockerInstallerPath)) {
    Write-Output "Docker installer not found at $dockerInstallerPath"
    exit 1
}

# Install Docker Desktop
Write-Output "Installing Docker Desktop..."

# Directly run the Docker installer command
& "$dockerInstallerPath" install --quiet --norestart

Write-Output "Docker Desktop installed successfully."
