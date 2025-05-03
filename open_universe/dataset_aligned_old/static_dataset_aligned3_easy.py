
# SPDX‑License‑Identifier: Apache‑2.0
"""
Dataset with optional pad‑to‑fixed‑length for training
and mask support for all splits.

Compatible with the old config that used
    audio_len: 4.0
If that key is present we treat it as `max_len_sec`.
"""
from __future__ import annotations
import os, logging, random
from pathlib import Path
from typing import Optional, Tuple, Union
import torch, torchaudio
from hydra.utils import to_absolute_path
import textgrid, math


log = logging.getLogger(__name__)

import re                                       # NEW



def reindex_blocks(blocks):
    out, cursor = [], 0
    for s_old, e_old, txt in blocks:
        length = e_old - s_old
        out.append((cursor, cursor + length, txt))
        cursor += length
    return out


# --- paste below over the old NoisyDataset -----------------------------
class NoisyDataset(torch.utils.data.Dataset):
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        audio_path: Union[str, Path],
        *,
        max_len_sec:  float | None = None,
        audio_len:    float | None = None,
        fixed_len_sec: float | None = None,
        fs:           int   = 16_000,
        split:        str   = "train",
        noisy_folder: str   = "noisy",
        clean_folder: str   = "clean",
        text_path:    str | None = None,
        part_used:    float = 1.0,
        # -------------- word‑cut parameters --------------------------- #
        win_len_sec:       float = 2.0,
        min_cut_sec_text:  float = 0.2,
        min_cut_sec_noise: float = 0.2,
        max_cut_sec:       float = 2.0,
        big_cut_min:       float = 0.75,
        # optional leading‑noise (disabled by default)
        starting_noise_min: float = 0.0,
        starting_noise_max: float = 0.0,
        textgrid_root:     str   | None = None,
        spacing_min:       float = 0.1, # backward compat only
        spacing_max:       float = 0.2, # backward compat only
        p_random:          float = 0.0, # backward compat only
        num_samples_audio: int = 1,     # backward compat only
    ):
        super().__init__()
        # ---------- original init code unchanged up to self.spacing_max ----------
        # (all the longer block is identical – shortened here for brevity)
        # ------------------------------------------------------------------------

        # backward‑compat alias
        if max_len_sec is None and audio_len is not None:
            max_len_sec = audio_len
        if max_len_sec is None:
            max_len_sec = 1e9            # effectively no limit

        self.fixed_len = int(fixed_len_sec * fs) if fixed_len_sec else None
        self.max_len   = int(max_len_sec  * fs)
        self.fs        = fs
        self.split     = split

        root = Path(to_absolute_path(str(audio_path))) / split
        self.noisy_path = root / noisy_folder
        self.clean_path = root / clean_folder
        self.clean_available = self.clean_path.exists()

        # --------------------------- file list ---------------------------
        files = sorted(os.listdir(self.noisy_path))
        if self.clean_available:
            files = sorted(set(files) & set(os.listdir(self.clean_path)))
        if part_used < 1.0:
            files = files[: max(1, int(len(files) * part_used))]

        # filter on length
        self.file_list, self.lengths = [], []
        for f in files:
            n = torchaudio.info(str(self.noisy_path / f)).num_frames
            if n <= self.max_len:
                self.file_list.append(f)
                self.lengths.append(n)

        self.text_path = Path(to_absolute_path(text_path)) if text_path else None
        log.info(
            f"[{split}] {len(self.file_list)} files (≤ {max_len_sec}s) "
            f"fixed_len={fixed_len_sec}"
        )
            
            
        # ---- cache word intervals for *all* splits -------------------
        self.tgdir = Path(to_absolute_path(str(textgrid_root))) / split if textgrid_root else None

        if self.tgdir:                     # ← build once, whatever the split
            self.win_N  = int(win_len_sec   * fs)
            # self.min_N  = int(min_cut_sec   * fs)
            self.min_text_N  = int(min_cut_sec_text  * fs)
            self.min_noise_N = int(min_cut_sec_noise * fs)
            self.min_N       = self.min_noise_N      # ← back-compat (leave rest of code untouched)

            self.max_N  = int(max_cut_sec   * fs)
            self.K      = num_samples_audio
            self.rng    = random.Random("wordcut")
            self.ivs    = []
            
            for fn in self.file_list:
                tg_path = self.tgdir / f"{Path(fn).stem}.TextGrid"
                if not tg_path.exists():               # ① no TextGrid → warn & continue
                    print(f"[dbg] TextGrid missing → words ignored for {fn}")
                    self.ivs.append([])                # keep index alignment
                    continue

                tg   = textgrid.TextGrid.fromFile(str(tg_path))
                tier = next((t for t in tg.tiers if "word" in t.name.lower()), None)
                if tier is None:                       # unlikely, but be safe
                    self.ivs.append([])
                    continue

                iv = [(int(i.minTime*fs),
                    int(i.maxTime*fs),
                    i.mark.strip())
                    for i in tier.intervals if i.mark.strip()]
                self.ivs.append(iv)
                
        # init other variables
        self.p_random = p_random
        self.big_cut_min = big_cut_min
        self.starting_noise_min = starting_noise_min
        self.starting_noise_max = starting_noise_max
        self.spacing_min = spacing_min
        self.spacing_max = spacing_max
        
        
        
    # ------------------------------------------------------------------ #
    def __len__(self): return len(self.file_list)
    # ------------------------------------------------------------------ #
    
    
        # -----------------------------------------------------------
    @staticmethod
    def _norm_txt(txt: str) -> str:
        """
        lower-case, remove punctuation, collapse multiple spaces
        examples:
            "Name, Matter!) "  ->  "name matter"
        """
        txt = txt.lower()
        txt = re.sub(r"[^a-z0-9 ]", " ", txt)       # keep letters/digits/space
        txt = re.sub(r"\s+", " ", txt)              # collapse 2+ spaces
        return txt.strip()


    def _load(self, p: Path) -> torch.Tensor:
        # wav, sr = torchaudio.load(p)
        # wav, sr = torchaudio.load(str(p))
        wav, sr = torchaudio.load(p, backend="soundfile")

        if sr != self.fs:
            wav = torchaudio.functional.resample(wav, sr, self.fs)
        return wav
    
    
    def __getitem__(self, idx):
        fn   = self.file_list[idx]
        noisy= self._load(self.noisy_path / fn)
        clean= self._load(self.clean_path / fn) if self.clean_available else torch.zeros_like(noisy)

        # ---------- validation / test branch (unchanged) ----------------
        if self.split != "train":
            txt  = ""
            if self.text_path and (self.text_path / f"{Path(fn).stem}.txt").exists():
                txt = self._norm_txt((self.text_path / f"{Path(fn).stem}.txt").read_text())
            mask = torch.ones(noisy.shape[-1])
            return noisy, clean, txt, mask
        # ----------------------------------------------------------------

        # -------------- TRAIN branch ------------------------------------
        fs, tgt_N = self.fs, self.win_N
        iv   = self.ivs[idx] if hasattr(self, "ivs") else []
        rng  = self.rng

        # ---- candidate spans (same as before) --------------------------
        spans=[]
        for i in range(len(iv)):
            for j in range(i, len(iv)):
                s, e = iv[i][0], iv[j][1]
                if self.min_text_N <= e - s <= self.max_N:
                    spans.append((s, e, " ".join(iv[k][2] for k in range(i, j + 1))))
                    
                    
        # ----------------------------------------------------------------
        # NEW: if the utterance is too short to satisfy min_text_N,
        # treat the **whole utterance** as one span so we keep its text.
        # ----------------------------------------------------------------
        if not spans and iv:
            full_txt = self._norm_txt(" ".join(w for *_, w in iv))
            spans.append((iv[0][0], iv[-1][1], full_txt))
        

        # ---- OPTIONAL leading noise (kept) -----------------------------
        chosen_src, blocks = [], []
        cursor = 0
        if self.starting_noise_max and spans:
            # pick a random duration and a random noise slice from before the first word
            st_noise_max = int(rng.uniform(self.starting_noise_min,
                                           self.starting_noise_max) * fs)
            if st_noise_max:
                ns, ne, _ = spans[0]
                take = min(st_noise_max, ns)
                if take:
                    chosen_src.append((ns - take, ns, ""))     # noise slice
                    blocks.append((0, take, ""))
                    cursor += take

        # ---- FIRST (largest) speech span ------------------------------
        if spans:
            big_thr = self.big_cut_min * tgt_N
            fit     = [c for c in spans if c[1] - c[0] <= tgt_N - cursor]
            big     = [c for c in fit if c[1] - c[0] >= big_thr]
            first   = rng.choice(big) if big else max(fit, key=lambda x: x[1]-x[0])
            chosen_src.append(first)
            blen   = first[1] - first[0]
            blocks.append((cursor, cursor + blen, first[2]))
            cursor += blen
            
        # ----------------------------------------------------------------
        # make sure we have at least ONE slice (fallback to start of file)
        # ----------------------------------------------------------------
        if not chosen_src:
            take = min(tgt_N, noisy.shape[-1])      # full utt or 2 s
            chosen_src.append((0, take, ""))
            blocks.append((0, take, ""))
            cursor = take
        

        # ---- assemble wav and pad the rest with zeros -----------------
        wav_c = torch.cat([clean[:, s:e] for s, e, _ in chosen_src], -1)
        wav_n = torch.cat([noisy[:, s:e] for s, e, _ in chosen_src], -1)

        remain = tgt_N - wav_n.shape[-1]
        mask   = torch.ones(tgt_N)
        if remain > 0:
            mask[-remain:] = 0.0
            pad = torch.zeros_like(wav_n[:, :remain])
            wav_c = torch.cat([wav_c, pad], -1)
            wav_n = torch.cat([wav_n, pad], -1)

        # ---- build text (only from the 1st span) ----------------------
        raw = " ".join(w for *_, w in reindex_blocks(blocks) if w)
        txt = self._norm_txt(raw)

        meta = {
            "clean_fn": str(self.clean_path / fn) if self.clean_available else "",
            "noisy_fn": str(self.noisy_path / fn),
            "blocks":   [(int(s), int(e), w) for s, e, w in chosen_src],
        }
        return wav_n, wav_c, txt, mask, meta
# -----------------------------------------------------------------------
