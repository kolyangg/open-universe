#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# sets up the “universe” conda environment on a fresh (or existing) machine
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="universe"
ENV_FILE="models/universe/environment_port.yaml"

# ----------------------------------------------------------------------------- #
# Initialise conda and enforce channel order
# ----------------------------------------------------------------------------- #
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. make sure conda-forge is *first* so its ready-made wheels win
conda config --add channels conda-forge 2>/dev/null || true
conda config --set channel_priority strict

# ----------------------------------------------------------------------------- #
# Create / update the environment
# ----------------------------------------------------------------------------- #
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists – updating it"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
    echo "Creating conda environment '${ENV_NAME}'"
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"
echo "Active conda environment: $CONDA_DEFAULT_ENV"

# ----------------------------------------------------------------------------- #
# Project install *without* re-installing dependencies that conda just resolved
# ----------------------------------------------------------------------------- #
python -m pip install --no-deps -e models/universe        # editable, no dep churn
python -m pip install --no-deps phonemizer                # same trick here

# ----------------------------------------------------------------------------- #
# Optional extras
# ----------------------------------------------------------------------------- #
# conda install -y -c conda-forge espeak-ng
# python -m pip install --no-deps py-espeak-ng

echo "✔  Setup finished — environment '${ENV_NAME}' is ready."
