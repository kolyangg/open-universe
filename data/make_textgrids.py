# make_textgrids.py
# Usage: python make_textgrids.py wav_dir txt_dir output_dir
# Requires MFA ≥3.2  (conda install -c conda-forge montreal-forced-aligner)
import sys, shutil, subprocess
from pathlib import Path
from tqdm import tqdm

wav_dir, txt_dir, out_dir = map(Path, sys.argv[1:4])
tmp_corpus = out_dir / "_corpus"
tmp_corpus.mkdir(parents=True, exist_ok=True)

# copy wav + lab
wavs = sorted(wav_dir.glob("*.wav"))
for w in tqdm(wavs, desc="Collecting corpus"):
    lab = txt_dir / f"{w.stem}.txt"
    if not lab.exists():
        print("Missing transcript:", lab)
        continue
    shutil.copy2(w,  tmp_corpus / w.name)
    shutil.copy2(lab, tmp_corpus / f"{w.stem}.lab")

# run MFA align (English)
subprocess.run([
    "mfa", "align", str(tmp_corpus),
    "english_us_arpa", "english_us_arpa",
    str(out_dir), "--clean", "--overwrite"
], check=True)
print("✓  TextGrids written to", out_dir)
