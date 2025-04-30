#!/bin/bash


# call this script
# models/universe/serv_setup.sh

# Exit on error
set -e

# ---------------------------------------------------------------------------
# Clone Miipher only if the folder is not there yet
# ---------------------------------------------------------------------------
if [ ! -d "_miipher" ]; then
    echo "Downloading _miipher repo"
    mkdir _miipher
    cd _miipher
    git clone https://github.com/ajaybati/miipher2.0.git
    cd ..
else
    echo "_miipher folder already exists – skipping download"
fi

# echo "Downloading and unzipping data..."
# models/universe/data/download.sh

echo "Downloading and unzipping data (running in background)..."
models/universe/data/download.sh &
DOWNLOAD_PID=$!

echo "Setting up environment..."
models/universe/setup_simple.sh

echo "Waiting for data download to complete..."
wait "$DOWNLOAD_PID"
echo "Data download finished."

# conda activate universe

if command -v conda &>/dev/null; then
    if conda env list | awk '{print $1}' | grep -q '^universe$'; then
        set +u
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate universe
        set -u
    else
        echo "Conda environment 'universe' not found. Please create it and rerun."
        exit 1
    fi
else
    echo "'conda' command not found. Install Miniconda/Anaconda first."
    exit 1
fi



echo "Preparing data..."
models/universe/data/prepare.sh


# set -euo pipefail

echo "Preparing text alignment data with MFA - train" 

WAV_DIR=data/voicebank_demand/16k/train/clean
TXT_DIR=data/voicebank_demand/trainset_28spk_txt
OUTPUT_DIR=data/voicebank_demand/textgrids/train

python models/universe/data/make_textgrids.py \
  "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR"


echo "Preparing text alignment data with MFA - val" 

WAV_DIR=data/voicebank_demand/16k/val/clean
TXT_DIR=data/voicebank_demand/trainset_28spk_txt
OUTPUT_DIR=data/voicebank_demand/textgrids/val

python models/universe/data/make_textgrids.py \
  "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR"

echo "Ready to go"