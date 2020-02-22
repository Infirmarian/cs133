#!/bin/bash
sudo apt update
sudo env DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo apt install build-essential

# install CUDA
echo 'deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /' |sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install cuda opencl-headers ocl-icd-opencl-dev -y --no-install-recommends

