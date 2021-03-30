# Installation and Usage Guide


## 1. Setup

- For exact reproducibility, every package dependencies are provided with docker.
- Ubuntu 18.04 and NVIDIA gpus are used.
- NVIDIA Graphic driver higher than 418 is required for CUDA 10.1 usage.

Check your graphic driver version:

```nvidia-smi```

## Upgrade NVIDIA-graphic driver if graphic driver lower than 418

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

apt-cache search nvidia | grep 418
# apt-cache search : print available NVIDIA driver list

sudo apt-get install nvidia-driver-418
# or : sudo apt install nvidia-418 

sudo reboot
# If conflicts with other program, please remove all related nvidia packges:
# sudo apt --purge autoremove nvidia*
# and retry installation
```


## 2. Install Docker

Refer to [official docker website](https://docs.docker.com/install/linux/docker-ce/ubuntu/):

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc # uninstall old docker version
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## 3. Install NVIDIA Container Toolkit

Refer to [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Try running docker

```docker run --gpus all -it --rm --shm-size 100G -v $PWD:/workspace  junhocho/hyperbolicgraphnn:8 nvidia-smi```

If some error occurs such as:

> usr/bin/docker-current: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?.

Run:

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

> `Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/containers/json: dial unix /var/run/docker.sock: connect: permission denied

Run: 

```bash
sudo usermod -a -G docker $USER
sudo chmod 666 /var/run/docker.sock
```

## 4. Docker-image description

Under cuda:10.1 with python3.6, these packages are installed via pip:

```bash
torch==1.2.0 torchvision==0.2.2
torch-scatter==1.4.0 torch-sparse==0.4.3 torch-cluster==1.4.5
torch-geometric==1.3.2 scikit-learn==0.20.0 networkx==2.2 numpy==1.16.2
```

