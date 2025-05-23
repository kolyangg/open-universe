#!/bin/bash

# Exit on error
set -euo pipefail               # safer bash

# ----------------------------------------------------------------
# Optional flag  --mamba | -m   ⇒ use mamba instead of conda
# ----------------------------------------------------------------
PKG="conda"
for arg in "$@"; do
  [[ $arg == "--mamba" || $arg == "-m" ]] && PKG="mamba"
done

cd ~/speech_enh

# echo "Create universe env"
# source activate base           # still fine; mamba shares base

echo "Create universe env"
if [[ $PKG == "mamba" ]]; then
  # ensure mamba exists in *base*
  if ! command -v mamba &>/dev/null; then
    conda install -n base -c conda-forge mamba -y
  fi
  eval "$(mamba shell hook -s bash)"
  mamba activate base
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate base
fi

# ---------------------------------------------------------------
# If user requested mamba and it is absent, install it once
# ---------------------------------------------------------------
if [[ "$PKG" == "mamba" && ! $(command -v mamba) ]]; then
    echo "mamba not found – installing into base environment ..."
    conda install -n base -c conda-forge mamba -y


    # # --- keep conda-libmamba-solver and libmambapy in sync ----------
    # # The “QueryFormat” error appears when the plugin is newer than
    # # the C++ library.  Update both to matching versions right now.
    # mamba install -n base -c conda-forge \
    #         "conda-libmamba-solver>=24.5" "libmambapy>=1.5.6" -y

    # # (optional but harmless) makes sure activate works in future shells
    # conda init bash


    # --- keep plugin + core lib matched (fixes QueryFormat error) ----
    install -n base -c conda-forge \
            "libmambapy>=1.5.6,<2" "conda-libmamba-solver>=24.5,<25" -y
    
    # re-enable the updated plugin for the running shell
    conda init --install-source mamba bash


fi


# $PKG create -n universe python=3.11.11 -y
# conda activate universe        # activation uses the usual conda hook

${PKG} create -n universe python=3.11.11 -y
if [[ $PKG == "mamba" ]]; then
  mamba activate universe
else
  conda activate universe
fi

echo "universe env created and activated"

echo "installing torch and pre-requirements"
pip install torch==2.5.1 torchvision torchaudio==2.5.1 --no-cache-dir

# sudo apt update
# sudo apt install -y nvidia-cuda-toolkit


# # ---------------------------------------------------------------
# # mamba run: be sure a C/C++ tool-chain is present for packages
# # that need compilation (e.g. pesq, cython-based libs)
# # ---------------------------------------------------------------
# if [[ "$PKG" == "mamba" ]]; then
#   sudo apt install -y build-essential        # gcc g++ make …
# fi


# System driver is already there on most hosts; add only compiler /
# runtime libraries that *don’t* ship NVML so they can’t clash.
$PKG install -y -c nvidia cuda-nvcc cudatoolkit-dev
# Defensive: strip any accidental NVML copies from the env
find "$CONDA_PREFIX/lib" -name 'libnvidia-ml.so*' -delete || true



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
