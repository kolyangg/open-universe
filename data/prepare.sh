#!/usr/bin/env bash
# Set bash to 'debug' mode:
# -e = exit on any error
# -u = treat undefined variables as errors
# -o pipefail = exit if any command in a pipeline fails
set -e
set -u
set -o pipefail

################################################################################
# Check & activate Conda environment 'universe' (new)
################################################################################
# if command -v conda &>/dev/null; then
#     if conda env list | awk '{print $1}' | grep -q '^universe$'; then

# Activate with whichever tool is available
if command -v mamba &>/dev/null && mamba env list | awk '{print $1}' | grep -q '^universe$'; then
    set +u
    eval "$(mamba shell hook -s bash)"
    mamba activate universe
    set -u
elif command -v conda &>/dev/null && conda env list | awk '{print $1}' | grep -q '^universe$'; then
   
        set +u
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate universe
        set -u

else
    # echo "'conda' command not found. Install Miniconda/Anaconda first."
    echo "Neither mamba nor conda with 'universe' env found."
    exit 1
fi


################################################################################
# Configuration
################################################################################
output_dir="./data/voicebank_demand"

################################################################################
# Prepare 48 kHz data
################################################################################
dir_48k="${output_dir}/48k"
if [ ! -d "${dir_48k}" ]; then
    echo "Preparing 48k data ..."
    mkdir -p "${dir_48k}"

    # ---------------------------
    # Train set: clean
    # ---------------------------
    mkdir -p "${dir_48k}/train/clean"
    cp -r "${output_dir}/clean_trainset_28spk_wav/"* "${dir_48k}/train/clean"

    # Create validation folder & move specific speaker files
    mkdir -p "${dir_48k}/val/clean"
    mv "${dir_48k}/train/clean"/{p226,p287}_*.wav "${dir_48k}/val/clean" 2>/dev/null || true

    # ---------------------------
    # Train set: noisy
    # ---------------------------
    mkdir -p "${dir_48k}/train/noisy"
    cp -r "${output_dir}/noisy_trainset_28spk_wav/"* "${dir_48k}/train/noisy"

    mkdir -p "${dir_48k}/val/noisy"
    mv "${dir_48k}/train/noisy"/{p226,p287}_*.wav "${dir_48k}/val/noisy" 2>/dev/null || true

    # ---------------------------
    # Test set
    # ---------------------------
    mkdir -p "${dir_48k}/test/clean"
    cp -r "${output_dir}/clean_testset_wav/"* "${dir_48k}/test/clean"

    mkdir -p "${dir_48k}/test/noisy"
    cp -r "${output_dir}/noisy_testset_wav/"* "${dir_48k}/test/noisy"
else
    echo "Already prepared 48k data"
fi

################################################################################
# Prepare 16 kHz and 24 kHz data via resampling
################################################################################
if [ ! -d "${output_dir}/16k" ]; then
    echo "Preparing 16k data ..."
    python -m open_universe.prepare.resample "${output_dir}/48k" "${output_dir}/16k" --fs 16000
else
    echo "Already prepared 16k data"
fi

if [ ! -d "${output_dir}/24k" ]; then
    echo "Preparing 24k data ..."
    python -m open_universe.prepare.resample "${output_dir}/48k" "${output_dir}/24k" --fs 24000
else
    echo "Already prepared 24k data"
fi

################################################################################
# Quick Checks
################################################################################
for sr in 16k 24k 48k; do
    for sub in clean noisy; do
        path_train="${output_dir}/${sr}/train/${sub}"
        n_files_train=$(find "${path_train}" -name '*.wav' | wc -l)
        if [ "${n_files_train}" -ne 10802 ]; then
            echo "Error: expected 10802 files in ${path_train}, found ${n_files_train}"
        fi

        path_val="${output_dir}/${sr}/val/${sub}"
        n_files_val=$(find "${path_val}" -name '*.wav' | wc -l)
        if [ "${n_files_val}" -ne 770 ]; then
            echo "Error: expected 770 files in ${path_val}, found ${n_files_val}"
        fi

        path_test="${output_dir}/${sr}/test/${sub}"
        n_files_test=$(find "${path_test}" -name '*.wav' | wc -l)
        if [ "${n_files_test}" -ne 824 ]; then
            echo "Error: expected 824 files in ${path_test}, found ${n_files_test}"
        fi
    done
done

echo "All done!"
