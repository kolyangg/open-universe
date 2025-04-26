#!/bin/bash

# Exit on error
set -e

cd ~/speech_enh

echo "Create universe env"
source activate base
conda create -n universe python=3.11.11 -y
conda activate universe
echo "universe env created and activated"

echo "installing torch and pre-requirements"
pip install torch==2.5.1 torchvision torchaudio==2.5.1

sudo apt update
sudo apt install -y nvidia-cuda-toolkit

conda install -y -c conda-forge gmpy2 numexpr


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
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic    english_us_arpa
mfa model download dictionary  english_us_arpa
