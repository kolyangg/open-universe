#!/bin/bash
# Exit on error
set -e

# Source the conda initialization script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if the environment "universe" exists.
if conda env list | awk '{print $1}' | grep -Fxq "universe"; then
    echo "Conda environment 'universe' already exists, skipping creation."
else
    echo "Creating conda environment 'universe'..."
    conda config --append channels conda-forge
    conda env create -f models/universe/environment_port.yaml
fi

# Activate the newly created environment
conda activate universe

echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Install
python -m pip install models/universe
pip install phonemizer

# conda install -c conda-forge espeak-ng
# pip install py-espeak-ng

echo "Active conda environment: $CONDA_DEFAULT_ENV"