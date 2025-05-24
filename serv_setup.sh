#!/usr/bin/env bash


# call this script
# models/universe/serv_setup.sh

# Exit on error
set -e


# ────────────────────────────────────────────────────────────────
# CLI flags
#   --mamba             → build/activate env with mamba
#   --script <cmd …>    → run arbitrary command after setup
# ────────────────────────────────────────────────────────────────
USE_MAMBA=0
SCRIPT_CMD=()            # array preserves quoted args
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--mamba)  USE_MAMBA=1; shift ;;
    --script)    shift; SCRIPT_CMD=("$@"); break ;;  # everything that follows
    *)           echo "Unknown flag $1"; exit 1 ;;
  esac
done

# record overall start
_t0=$(date +%s%N)           # nanoseconds since epoch

# helper → duration in seconds to 2 d.p.
_sec() { awk "BEGIN{printf \"%.2f\", ($2-$1)/1000000000}"; }

# # ────────────────────────────────────────────────────────────────
# # CLI:  -m / --mamba   ⇒ use mamba inside setup_simple.sh
# # ────────────────────────────────────────────────────────────────
# USE_MAMBA=0
# for arg in "$@"; do
#   [[ $arg == "-m" || $arg == "--mamba" ]] && USE_MAMBA=1
# done

# ────────────────────────────────────────────────────────────────
# Ask once whether to generate MFA TextGrids later
# ────────────────────────────────────────────────────────────────
read -rp "Generate MFA TextGrids after data prep? [y/N]: " GEN_TG



# ────────────────────────────────────────────────────────────────
# 1. Ask for WandB key (blank = skip) – once at startup
# ────────────────────────────────────────────────────────────────
read -rsp "Weights & Biases API key (press Enter to skip): " WANDB_API_KEY
echo


### Install mamba solver if not already installed
if ! command -v mamba &>/dev/null; then
  echo "Installing mamba solver..."
  if command -v conda &>/dev/null; then
    conda install -n base -c conda-forge -y \
        "mamba>=1.5.6" \
        "libmambapy>=1.5.6,<2" \
        "conda-libmamba-solver>=24.5,<25"
  else
    echo "Conda not found. Please install Miniconda/Anaconda first."
    exit 1
  fi
else
  echo "Mamba solver already installed."
fi


# ---------------------------------------------------------------------------
# Clone Miipher only if the folder is not there yet
# ---------------------------------------------------------------------------
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
# if conda env list | awk '{print $1}' | grep -q '^universe$'; then

# 2. Env: create / reuse depending on user choice
ENV_CMD=$([[ $USE_MAMBA -eq 1 ]] && echo mamba || echo conda)
if $ENV_CMD env list | awk '{print $1}' | grep -q '^universe$'; then

  read -rp "Conda env 'universe' already exists. Re-install? [y/N]: " REPLY
  [[ "$REPLY" =~ ^[Yy]$ ]] && RUN_SETUP=1 || RUN_SETUP=0
else
  RUN_SETUP=1
fi

if [[ $RUN_SETUP -eq 1 ]]; then
  env_s=$(date +%s%N)
  echo "Setting up environment..."
  # if [[ $USE_MAMBA -eq 1 ]]; then
  #   models/universe/setup_simple.sh --mamba
  # else
  #   models/universe/setup_simple.sh
  # fi

  [[ $USE_MAMBA -eq 1 ]] \
      && models/universe/setup_simple2.sh --mamba \
      || models/universe/setup_simple2.sh


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

# if command -v conda &>/dev/null; then
#     if conda env list | awk '{print $1}' | grep -q '^universe$'; then
#         set +u
#         # shellcheck disable=SC1091
#         source "$(conda info --base)/etc/profile.d/conda.sh"
#         conda activate universe
#         set -u

if command -v "$ENV_CMD" &>/dev/null; then
    if $ENV_CMD env list | awk '{print $1}' | grep -q '^universe$'; then
        set +u
        # if [[ $USE_MAMBA -eq 1 ]]; then
        #     eval "$(mamba shell hook -s bash)"
        #     mamba activate universe
        # else
        #     source "$(conda info --base)/etc/profile.d/conda.sh"
        #     conda activate universe
        # fi

        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate universe
        set -u


    else
        echo "'universe' env not found under $ENV_CMD. Please create it and rerun."
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
models/universe/data/prepare2.sh
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
printf "Repo clone:           %6s\n" "$(_sec $clone_s $clone_e)"
printf "Data download:        %6s\n" "$(_sec $dl_s $dl_e)"
printf "Env setup (%s):       %6s\n" "$ENV_CMD" "$(_sec $env_s $env_e)"
printf "Data prepare:         %6s\n" "$(_sec $prep_s $prep_e)"
if [[ $GEN_TG =~ ^[Yy]$ ]]; then
  printf "TextGrid prepare:     %6s\n" "$(_sec $tg_s $tg_e)"
fi
_t1=$(date +%s%N)
printf "TOTAL:                %6s\n" "$(_sec $_t0 $_t1)"

echo "Ready to go"


# ────────────────────────────────────────────────────────────────
# Optional follow-up command (e.g. start training)
# ────────────────────────────────────────────────────────────────
if ((${#SCRIPT_CMD[@]})); then
  echo -e "\nRunning post-setup command:\n ${SCRIPT_CMD[*]}"
  "${SCRIPT_CMD[@]}"
fi