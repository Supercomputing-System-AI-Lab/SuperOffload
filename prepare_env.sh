# Set BASE_DIR environment variable to current directory for later reference
echo 'export BASE_DIR="'$PWD'"' >> ~/.bashrc

# Download and install Miniconda for Python package management
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
rm Miniconda3-latest-Linux-aarch64.sh
source ~/.bashrc

# Clone the main SuperOffload repository
git clone https://github.com/Supercomputing-System-AI-Lab/SuperOffload.git
cd SuperOffload
# Set SUPEROFFLOAD_DIR environment variable to this directory
echo 'export SUPEROFFLOAD_DIR="'$PWD'"' >> ~/.bashrc
cd $BASE_DIR

# Clone the modified Megatron-DeepSpeed repository
git clone https://github.com/Supercomputing-System-AI-Lab/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
echo 'export MEGATRON_DIR="'$PWD'"' >> ~/.bashrc
# Switch to the superoffload-specific branch
git checkout superofflaod
cd $BASE_DIR

# Clone the modified DeepSpeed repository
git clone https://github.com/Supercomputing-System-AI-Lab/DeepSpeed.git
cd DeepSpeed
echo 'export DEEPSPEED_DIR="'$PWD'"' >> ~/.bashrc
# Switch to the superoffload-specific branch
git checkout superoffload
# Install DeepSpeed in development mode
# Note: Installation may encounter issues at this step; if so, it can be installed later
cd $BASE_DIR

# Reload environment variables
source ~/.bashrc
# Create conda environment from the provided YAML file
conda env create -f $BASE_DIR/SuperOffload/environment.yaml
# Activate the newly created environment
conda activate ae

# Load CUDA module (version 12.6.1)
module load cuda/12.6.1
# Install PyTorch with CUDA 12.6 support from test repository
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0   --index-url https://download.pytorch.org/whl/test/cu126
pip install pybind11

# Install specific GCC version for compatibility
conda install -c conda-forge gcc==11.4.0 gxx_linux-aarch64
# Set compiler environment variables for ARM64 architecture
export CC=aarch64-conda-linux-gnu-gcc
export CXX=aarch64-conda-linux-gnu-g++

# Install NVIDIA Apex for mixed precision training
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 25.07
# Install Apex with C++ and CUDA extensions
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

cd $DEEPSPEED_DIR
pip install -e .
# Potentional Issue 1
# File ".../envs/ds/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2016, in <listcomp>
# supported_sm = [int(arch.split('_')[1]) for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
# "ValueError: invalid literal for int() with base 10: '90a'"

# Solution 1
# modify the line to exclude architectures ending with 'a'.
# vim ~/miniconda3/envs/ae/lib/python3.11/site-packages/torch/utils/cpp_extension.py
# supported_sm = [int(arch.split('_')[1]) for arch in torch.cuda.get_arch_list() if 'sm_' in arch and not arch.endswith('a')]
