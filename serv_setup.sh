#!/bin/bash


# call this script
# models/universe/serv_setup.sh

# Exit on error
set -e

# record overall start
_t0=$(date +%s%N)           # nanoseconds since epoch

# helper → duration in seconds to 2 d.p.
_sec() { awk "BEGIN{printf \"%.2f\", ($2-$1)/1000000000}"; }

# ────────────────────────────────────────────────────────────────
# CLI:  -m / --mamba   ⇒ use mamba inside setup_simple.sh
# ────────────────────────────────────────────────────────────────
USE_MAMBA=0
for arg in "$@"; do
  [[ $arg == "-m" || $arg == "--mamba" ]] && USE_MAMBA=1
done

# ────────────────────────────────────────────────────────────────
# Ask once whether to generate MFA TextGrids later
# ────────────────────────────────────────────────────────────────
read -rp "Generate MFA TextGrids after data prep? [y/N]: " GEN_TG



# ────────────────────────────────────────────────────────────────
# 1. Ask for WandB key (blank = skip) – once at startup
# ────────────────────────────────────────────────────────────────
read -rsp "Weights & Biases API key (press Enter to skip): " WANDB_API_KEY
echo

# ---------------------------------------------------------------------------
# Clone Miipher only if the folder is not there yet
# ---------------------------------------------------------------------------
if [ ! -d "_miipher" ]; then
    clone_s=$(date +%s%N)
    if [ ! -d "_miipher" ]; then

    echo "Downloading _miipher repo"
    mkdir _miipher
    cd _miipher
    git clone https://github.com/ajaybati/miipher2.0.git
    cd ..
else
    echo "_miipher folder already exists – skipping download"
fi
clone_e=$(date +%s%N)

# echo "Downloading and unzipping data..."
# models/universe/data/download.sh

echo "Downloading and unzipping data (running in background)..."
dl_s=$(date +%s%N)
models/universe/data/download.sh &
DOWNLOAD_PID=$!

# 2. Conda env: create / reuse depending on user choice
if conda env list | awk '{print $1}' | grep -q '^universe$'; then
  read -rp "Conda env 'universe' already exists. Re-install? [y/N]: " REPLY
  [[ "$REPLY" =~ ^[Yy]$ ]] && RUN_SETUP=1 || RUN_SETUP=0
else
  RUN_SETUP=1
fi

if [[ $RUN_SETUP -eq 1 ]]; then
  env_s=$(date +%s%N)
  echo "Setting up environment..."
  if [[ $USE_MAMBA -eq 1 ]]; then
    models/universe/setup_simple.sh --mamba
  else
    models/universe/setup_simple.sh
  fi
else
  echo "Using existing 'universe' environment."
fi
env_e=$(date +%s%N)


echo "Waiting for data download to complete..."
wait "$DOWNLOAD_PID"
dl_e=$(date +%s%N)
echo "Data download finished."


# ────────────────────────────────────────────────────────────────
# 3. Activate env and log in to WandB (if key provided)
# ────────────────────────────────────────────────────────────────

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

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "Logging in to Weights & Biases..."
  wandb login --relogin "${WANDB_API_KEY}"
else
  echo "No WandB key supplied – skipping login."
fi


echo "Preparing data..."
prep_s=$(date +%s%N)
models/universe/data/prepare.sh
prep_e=$(date +%s%N)


# set -euo pipefail


# ────────────────────────────────────────────────────────────────
# Run TextGrid maker only if the user opted in
# ────────────────────────────────────────────────────────────────
if [[ $GEN_TG =~ ^[Yy]$ ]]; then
  tg_s=$(date +%s%N)
  models/universe/data/prepare_textgrids.sh
  tg_e=$(date +%s%N)
fi


# ────────────────────────────────────────────────────────────────
# Timing summary
# ────────────────────────────────────────────────────────────────
echo -e "\n----- Timing summary (seconds) -----"
printf "Repo clone:           %6s\n"   "$(_sec $clone_s $clone_e)"
printf "Data download:        %6s\n"   "$(_sec $dl_s    $dl_e)"
printf "Env setup (%s): %6s\n" \
        "$([[ $USE_MAMBA -eq 1 ]] && echo mamba || echo conda)" \
        "$(_sec $env_s   $env_e)"
printf "Data prepare:         %6s\n"   "$(_sec $prep_s  $prep_e)"
if [[ $GEN_TG =~ ^[Yy]$ ]]; then
  printf "TextGrid prepare:     %6s\n"   "$(_sec $tg_s    $tg_e)"
fi
_t1=$(date +%s%N)
printf "TOTAL:                %6s\n"   "$(_sec $_t0 $_t1)"

echo "Ready to go"
