#!/usr/bin/env bash
set -euo pipefail               # safer bash

# ──────────────────────────────────────────────────────────────
# Optional flag  --mamba | -m ⇒ use mamba instead of conda
# ──────────────────────────────────────────────────────────────
PKG=conda
for arg in "$@"; do
  [[ $arg == "--mamba" || $arg == "-m" ]] && PKG=mamba
done

cd ~/speech_enh

echo "▶ Create / activate base env"
if [[ $PKG == mamba ]]; then
  # be sure mamba exists in *base*
  if ! command -v mamba &>/dev/null; then
    conda install -n base -c conda-forge -y mamba
  fi
  eval "$(mamba shell hook -s bash)"
  mamba activate base
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate base
fi

# ──────────────────────────────────────────────────────────────
# Keep the libmamba stack coherent (fixes “QueryFormat” crash)
# ──────────────────────────────────────────────────────────────
${PKG} install -n base -c conda-forge -y \
        "mamba>=1.5.6" \
        "libmambapy>=1.5.6,<2" \
        "conda-libmamba-solver>=24.5,<25"
# prefer the (working) libmamba solver, fall back silently if absent
conda config --set solver libmamba 2>/dev/null || conda config --set solver classic

# ──────────────────────────────────────────────────────────────
# Create the project env
# ──────────────────────────────────────────────────────────────
echo "▶ Create universe env"
${PKG} create -n universe python=3.11.11 -y
[[ $PKG == mamba ]] && mamba activate universe || conda activate universe
echo "universe env created and activated"

echo "▶ installing torch"
pip install --no-cache-dir \
     torch==2.5.1 \
     torchvision \
     torchaudio==2.5.1

# ──────────────────────────────────────────────────────────────
# CUDA build tools **without** touching the host driver
# * cudatoolkit-dev   → headers, static libs (conda-forge)
# * cuda-nvcc         → nvcc compiler       (nvidia)
# ──────────────────────────────────────────────────────────────
echo "▶ installing CUDA headers & nvcc into env"
${PKG} install -y -c conda-forge -c nvidia \
        cudatoolkit-dev \
        cuda-nvcc

# Defensive: never let the env shadow the running NVML library
find "$CONDA_PREFIX/lib" -maxdepth 1 -name 'libnvidia-ml.so*' -delete || true

# math libs needed by requirements
${PKG} install -y -c conda-forge gmpy2 numexpr

echo "▶ installing Python requirements"
pip install --no-cache-dir -r models/universe/univ_requirements.txt

echo "▶ installing project in editable mode + extras"
python -m pip install --no-deps -e models/universe
pip install onnxruntime textgrid

# ──────────────────────────────────────────────────────────────
# System packages (no driver – avoids NVML mismatch)
# ──────────────────────────────────────────────────────────────
echo "▶ installing auxiliary system tools"
sudo apt-get update -qq
sudo apt-get install -y unzip build-essential  # gcc, g++, make …

# ──────────────────────────────────────────────────────────────
# Montreal Forced Aligner
# ──────────────────────────────────────────────────────────────
echo "▶ installing MFA & downloading English models"
${PKG} install -y -c conda-forge montreal-forced-aligner
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa

echo -e "\n✅  setup_simple.sh finished successfully"
