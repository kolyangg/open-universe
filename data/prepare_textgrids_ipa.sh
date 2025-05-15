#!/usr/bin/env bash
set -e

echo "Preparing text alignment data with MFA - train (IPA)"

WAV_DIR=data/voicebank_demand/16k/train/clean
TXT_DIR=data/voicebank_demand/trainset_28spk_txt
OUTPUT_DIR=data/voicebank_demand/textgrids_ipa/train

if [ -d "$OUTPUT_DIR" ]; then
  read -rp "TextGrids for TRAIN already exist. Re-generate? [y/N]: " R
  [[ $R =~ ^[Yy]$ ]] && python models/universe/data/make_textgrids_ipa.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR" \
                     || echo "Skipping train TextGrids."
else
  python models/universe/data/make_textgrids_ipa.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR"
fi

echo "Preparing text alignment data with MFA - val"

WAV_DIR=data/voicebank_demand/16k/val/clean
TXT_DIR=data/voicebank_demand/trainset_28spk_txt
OUTPUT_DIR=data/voicebank_demand/textgrids_ipa/val

if [ -d "$OUTPUT_DIR" ]; then
  read -rp "TextGrids for VAL already exist. Re-generate? [y/N]: " R
  [[ $R =~ ^[Yy]$ ]] && python models/universe/data/make_textgrids_ipa.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR" \
                     || echo "Skipping val TextGrids."
else
  python models/universe/data/make_textgrids_ipa.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR"
fi

echo "TextGrid preparation finished."
