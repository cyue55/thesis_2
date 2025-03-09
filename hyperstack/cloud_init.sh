#!/bin/bash
exec 1>log.out 2>&1

# Set DEBIAN_FRONTEND to noninteractive
export DEBIAN_FRONTEND=noninteractive

# Install Docker
if [ ! -x "$(command -v docker)" ]; then
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# Install GnuPG
if [ ! -x "$(command -v gpg)" ]; then
    sudo apt-get update
    sudo apt-get install -y gnupg
fi

# Install NVIDIA Container Toolkit
if [ ! -x "$(command -v nvidia-container-cli)" ]; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
fi

# Clone repo
if [ ! -d "mbchl" ]; then
    git clone "https://foo:${GITLAB_ACCESS_TOKEN}@hea-gitlab.healthtech.dtu.dk/phigon/mbchl.git"
fi

# Run Docker container
sudo docker run --gpus all -v $(pwd)/mbchl:/mbchl --rm \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_PROJECT="${WANDB_PROJECT}" \
    -e WANDB_ENTITY="${WANDB_ENTITY}" \
    -e AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL}" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    --shm-size=16G \
    -w "/mbchl/recipes/${MBCHL_RECIPE}" \
    philgzl/mbchl:latest \
    python train.py "models/${MBCHL_MODEL}"

# Delete VM
sudo docker run -v $(pwd)/mbchl:/mbchl --rm \
    -e HYPERSTACK_API_KEY="${HYPERSTACK_API_KEY}" \
    -w /mbchl/hyperstack \
    philgzl/mbchl:latest \
    python hyperstack_cli.py delete_vm_by_name "${HYPERSTACK_VM_NAME}"
