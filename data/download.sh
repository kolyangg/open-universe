#!/usr/bin/env bash
# Set bash to 'debug' mode:
# -e = exit on any error
# -u = treat undefined variables as errors
# -o pipefail = exit if any command in a pipeline fails
set -e
set -u
set -o pipefail

################################################################################
# Configuration
################################################################################
output_dir="./data/voicebank_demand"
mkdir -p "${output_dir}"

# URLs for the dataset
# Refer to https://datashare.ed.ac.uk/handle/10283/2791
LINKS=(
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_28spk_txt.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"
    "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"
)

################################################################################
# Make sure unzip exists (new)
################################################################################
if ! command -v unzip &>/dev/null; then
    echo "'unzip' not found – installing ..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y unzip
    elif command -v yum &>/dev/null; then
        sudo yum install -y unzip
    else
        echo "Could not install 'unzip' automatically. Please install it and rerun."
        exit 1
    fi
fi

################################################################################
# Download (now in parallel) & Unzip
################################################################################
pids=()
for link in "${LINKS[@]}"; do
    name=$(basename "${link}")
    stem="${name%.*}"

    if [ ! -d "${output_dir}/${stem}" ]; then
        echo "Downloading ${name} ..."
        wget --continue "${link}" -P "${output_dir}" &
        pids+=($!)
    else
        echo "Already unzipped ${name}"
    fi
done

# Wait for all background downloads to finish
for pid in "${pids[@]}"; do wait "${pid}"; done

# Unzip (serial – keeps original behaviour)
for link in "${LINKS[@]}"; do
    name=$(basename "${link}")
    stem="${name%.*}"

    if [ ! -d "${output_dir}/${stem}" ]; then
        echo "Unzipping ${name} ..."
        unzip -q "${output_dir}/${name}" -d "${output_dir}"
    fi
done

# Remove Mac-specific hidden folders if they appear
rm -rf "${output_dir}/__MACOSX"

echo "Download & unzip completed!"
