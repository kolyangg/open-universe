#!/bin/bash

# Exit on error
set -e

# echo "Downloading main repo"
# git clone --recursive https://github.com/kolyangg/speech_enh.git
# cd speech_enh
# this script
# 

echo "Downloading _miipher repo"
mkdir _miipher
cd _miipher
git clone https://github.com/ajaybati/miipher2.0.git
cd ..

echo "Setting up environment..."
models/universe/setup_simple.sh

echo "Downloading data..."
python3 utils/dataset_download.py

echo "Ready to go"