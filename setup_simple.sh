#!/bin/bash

# Exit on error
set -e

# ----------------------------------------------------------------
# Optional flag  --mamba | -m   ⇒ use mamba instead of conda
# ----------------------------------------------------------------
PKG="conda"
for arg in "$@"; do
  [[ $arg == "--mamba" || $arg == "-m" ]] && PKG="mamba"
done

cd ~/speech_enh

echo "Create universe env"
source activate base           # still fine; mamba shares base
$PKG create -n universe python=3.11.11 -y
conda activate universe        # activation uses the usual conda hook

echo "universe env created and activated"

echo "installing torch and pre-requirements"
pip install torch==2.5.1 torchvision torchaudio==2.5.1 --no-cache-dir

sudo apt update
sudo apt install -y nvidia-cuda-toolkit

$PKG install -y -c conda-forge gmpy2 numexpr
$PKG install -y nvidia::cuda-nvcc # for NVCC v12

#### Install requirements
echo "installing requirements.txt"
# pip install -r models/universe/univ_requirements.txt
pip install -r models/universe/univ_requirements.txt --no-cache-dir


### Other installations (separate)
echo "installing other libraries"
python -m pip install --no-deps -e models/universe 
pip install onnxruntime
pip install textgrid

### Install unzip
echo "unzip"
sudo apt update
sudo apt install unzip

### Install MFA
echo "installing MFA for text alignment"
${PKG} install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic    english_us_arpa
mfa model download dictionary  english_us_arpa
