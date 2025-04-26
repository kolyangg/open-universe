#!/bin/bash

# Exit on error
set -e

# echo "Downloading main repo"
# git clone --recursive https://github.com/kolyangg/speech_enh.git
# cd speech_enh
# cd models/universe
# git checkout main
# cd ~/speech_enh


# call this script
# models/universe/serv_setup.sh

echo "Downloading _miipher repo"
mkdir _miipher
cd _miipher
git clone https://github.com/ajaybati/miipher2.0.git
cd ..

echo "Setting up environment..."
models/universe/setup_simple.sh

conda activate universe

echo "Downloading data..."
models/universe/data/prepare_voicebank_demand.sh

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