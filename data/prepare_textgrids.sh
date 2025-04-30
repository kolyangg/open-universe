#!/usr/bin/env bash
set -e

echo "Preparing text alignment data with MFA - train"

WAV_DIR=data/voicebank_demand/16k/train/clean
TXT_DIR=data/voicebank_demand/trainset_28spk_txt
OUTPUT_DIR=data/voicebank_demand/textgrids/train

if [ -d "$OUTPUT_DIR" ]; then
  read -rp "TextGrids for TRAIN already exist. Re-generate? [y/N]: " R
  [[ $R =~ ^[Yy]$ ]] && python models/universe/data/make_textgrids.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR" \
                     || echo "Skipping train TextGrids."
else
  python models/universe/data/make_textgrids.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR"
fi

echo "Preparing text alignment data with MFA - val"

WAV_DIR=data/voicebank_demand/16k/val/clean
TXT_DIR=data/voicebank_demand/trainset_28spk_txt
OUTPUT_DIR=data/voicebank_demand/textgrids/val

if [ -d "$OUTPUT_DIR" ]; then
  read -rp "TextGrids for VAL already exist. Re-generate? [y/N]: " R
  [[ $R =~ ^[Yy]$ ]] && python models/universe/data/make_textgrids.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR" \
                     || echo "Skipping val TextGrids."
else
  python models/universe/data/make_textgrids.py "$WAV_DIR" "$TXT_DIR" "$OUTPUT_DIR"
fi

echo "TextGrid preparation finished."
