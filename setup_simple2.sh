#!/usr/bin/env bash
# ------------------------------------------------------------------
# Build the “universe” Conda/Mamba environment without breaking the
# host NVIDIA driver or the libmamba solver stack.
# ------------------------------------------------------------------
set -e -o pipefail          # no nounset (set -u) ⇒ avoid MAMBA_ROOT_PREFIX errors

# Use classic solver until we finish upgrading the libmamba trio
export CONDA_SOLVER=classic

# ──────────────────────────────────────────────────────────────
# Parse optional flag  --mamba | -m  (default = conda)
# ──────────────────────────────────────────────────────────────
PKG=conda
for arg in "$@"; do
  [[ $arg == "--mamba" || $arg == "-m" ]] && PKG=mamba
done

cd ~/speech_enh

echo "▶ Create / activate base env"
if [[ $PKG == mamba ]]; then
  # ensure mamba exists in *base*
  if ! command -v mamba &>/dev/null; then
    CONDA_SOLVER=classic conda install -n base -c conda-forge -y mamba
  fi
  # the shell hook creates MAMBA_ROOT_PREFIX – nounset must stay OFF
  eval "$(mamba shell hook -s bash)"
  mamba activate base
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate base
  conda config --add channels defaults    # ← silences the FutureWarning
fi

# ──────────────────────────────────────────────────────────────
# Bring the libmamba stack to matching versions (fixes QueryFormat)
# ──────────────────────────────────────────────────────────────
${PKG} install -n base -c conda-forge -y \
        "mamba>=1.5.6" \
        "libmambapy>=1.5.6,<2" \
        "conda-libmamba-solver>=24.5,<25"

# libmamba is now healthy – make it the default and drop override
conda config --set solver libmamba
unset CONDA_SOLVER

# ──────────────────────────────────────────────────────────────
# Create the project env
# ──────────────────────────────────────────────────────────────
echo "▶ Create universe env"
${PKG} create -n universe python=3.11.11 -y
[[ $PKG == mamba ]] && mamba activate universe || conda activate universe
echo "universe env created and activated"

echo "▶ installing PyTorch stack"
pip install --no-cache-dir \
     torch==2.5.1 \
     torchvision \
     torchaudio==2.5.1

# ──────────────────────────────────────────────────────────────
# CUDA build tools (headers + nvcc) – no driver libraries
# ──────────────────────────────────────────────────────────────
echo "▶ installing CUDA headers & nvcc into env"
${PKG} install -y -c conda-forge -c nvidia \
        cudatoolkit-dev \
        cuda-nvcc

# Remove any stray NVML copy to avoid driver/library mismatch
find "$CONDA_PREFIX/lib" -maxdepth 1 -name 'libnvidia-ml.so*' -delete || true

# math libs needed by requirements
${PKG} install -y -c conda-forge gmpy2 numexpr

echo "▶ installing Python requirements"
pip install --no-cache-dir -r models/universe/univ_requirements.txt

echo "▶ installing project in editable mode + extras"
python -m pip install --no-deps -e models/universe
pip install onnxruntime textgrid

# ──────────────────────────────────────────────────────────────
# System packages (no GPU driver – prevents NVML mismatch)
# ──────────────────────────────────────────────────────────────
echo "▶ installing auxiliary system tools"
sudo apt-get update -qq
sudo apt-get install -y unzip build-essential  # gcc, g++, make …

# ──────────────────────────────────────────────────────────────
# rclone – cloud‑sync helper (needed by upload/download tools)
# ──────────────────────────────────────────────────────────────
echo "▶ checking rclone"
if ! command -v rclone &>/dev/null; then
  echo "   rclone not found – installing via conda‑forge"
  ${PKG} install -y -c conda-forge rclone
else
  echo "   rclone already present ($(rclone --version | head -1))"
fi

# ──────────────────────────────────────────────────────────────
# Montreal Forced Aligner
# ──────────────────────────────────────────────────────────────
echo "▶ installing MFA & downloading English models"
${PKG} install -y -c conda-forge montreal-forced-aligner
mfa model download acoustic   english_us_arpa
mfa model download dictionary english_us_arpa

echo -e "\n✅  setup_simple2.sh finished successfully"
