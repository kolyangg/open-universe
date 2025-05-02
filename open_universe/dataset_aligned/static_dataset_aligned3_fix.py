
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



class NoisyDataset(torch.utils.data.Dataset):
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        audio_path: Union[str, Path],
        *,
        max_len_sec:  float | None = None,      # filter ≥ this length
        audio_len:    float | None = None,      # ← alias for max_len_sec
        fixed_len_sec: float | None = None,     # pad‑to‑len (train only)
        fs:           int   = 16_000,
        split:        str   = "train",
        noisy_folder: str   = "noisy",
        clean_folder: str   = "clean",
        text_path:    str | None = None,
        part_used:    float = 1.0,
        # -------------- word-cut parameters (new) ------------------- #
        win_len_sec:       float = 2.0,
        # min_cut_sec:       float = 0.2,
        min_cut_sec_text:  float = 0.2,
        min_cut_sec_noise: float = 0.2,
        max_cut_sec:       float = 2.0,
        num_samples_audio: int   = 1,
        p_random:          float = 1.0,
        big_cut_min:       float = 0.75,
        starting_noise_min: float = 0.0,
        starting_noise_max: float = 0.2,
        spacing_min:       float = 0.1,
        spacing_max:       float = 0.2,
        textgrid_root:     str   | None = None,
    ):
        super().__init__()

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

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        fn   = self.file_list[idx]
        noisy= self._load(self.noisy_path / fn)
        clean= self._load(self.clean_path / fn) if self.clean_available else torch.zeros_like(noisy)
        
        
        # ---------- simple, no-augmentation path for val/test -------------
        if self.split != "train":
            txt = ""
            # if self.text_path and (self.text_path / f"{Path(fn).stem}.txt").exists():
            #     txt = (self.text_path / f"{Path(fn).stem}.txt").read_text().strip().lower()
            if self.text_path and (self.text_path / f"{Path(fn).stem}.txt").exists():
                txt = self._norm_txt( (self.text_path / f"{Path(fn).stem}.txt").read_text() )   # ▼


            mask = torch.ones(noisy.shape[-1])           # everything is valid
            return noisy, clean, txt, mask



        # -------- variables always needed in training ---------------------
        # make sure they exist even when there is no TextGrid folder
        if not hasattr(self, "win_N"):
            self.win_N  = int(self.fixed_len or 2.0 * self.fs)  # default 2 s
            # self.min_N  = int(0.2 * self.fs)
            self.max_N  = int(2.0 * self.fs)
            self.min_text_N   = int(0.2 * self.fs)
            self.min_noise_N  = self.min_text_N
            self.min_N        = self.min_noise_N

        # always fetch intervals – may be empty
        iv = self.ivs[idx] if hasattr(self, "ivs") else []

        fs     = self.fs
        rng    = self.rng                           # stateful rng only for training
        tgt_N  = self.win_N
        min_N  = self.min_N
        max_N  = self.max_N

        # --- candidate multi-word spans ------------------------------
        spans=[]
        n=len(iv)
        for i in range(n):
            for j in range(i,n):
                s,e=iv[i][0],iv[j][1]
                if min_N<=e-s<=max_N:
                    spans.append((s,e," ".join(iv[k][2] for k in range(i,j+1))))

        # --- helper: pick noise slices --------------------------------
        wav_len=noisy.shape[-1]
        leading =(0,iv[0][0]) if iv else (0,wav_len)
        trailing=(iv[-1][1],wav_len) if iv else (0,0)
        noise_blocks=[]
        for ns,ne in (leading,trailing):
            for st in range(ns,ne,max_N):
                en=min(st+max_N,ne)
                if en-st>=min_N: noise_blocks.append((st,en))
        
        # if no suitable noise blocks exist, fall back to a dummy 1-sample block
        # (it will be padded in step 3)
        if not noise_blocks:
               noise_blocks.append((0, 1))

        # # --- window assembly (mirrors make_debug_set) -----------------
        # remaining=tgt_N; cursor=0
        # chosen_src=[]; blocks=[]; mask=torch.ones(tgt_N)


        # # --- window assembly (mirrors make_debug_set) -----------------
        # remaining = tgt_N
        # cursor    = 0
        # chosen_src, blocks = [], []
        # mask = torch.ones(tgt_N)

        # # optional starting noise
        # start_N = int(rng.uniform(self.starting_noise_min,
        #                         self.starting_noise_max) * fs)
        # if start_N and remaining >= start_N and noise_blocks:
        
        
        # --- window assembly (mirrors make_debug_set) -----------------
        remaining = tgt_N
        cursor    = 0
        chosen_src, blocks = [], []
        mask = torch.ones(tgt_N)

        # ---------------------------------------------------------------
        # 1) decide if we are in “short-utterance” mode
        # ---------------------------------------------------------------
        short_utt = noisy.shape[-1] <= tgt_N

        # optional starting noise → **skip for short utterances**
        start_N = 0
        if not short_utt:
            start_N = int(rng.uniform(self.starting_noise_min,
                                       self.starting_noise_max) * fs)

        if start_N and remaining >= start_N and noise_blocks:

            nb = rng.choice(noise_blocks)              # pick one candidate
            avail = nb[1] - nb[0]

            if avail <= start_N:                       # block shorter than demand
                st, take = nb[0], avail                #   → just take the whole block
            else:                                      # block long enough
                st   = rng.randint(nb[0], nb[1] - start_N)
                take = start_N

            chosen_src.append((st, st + take, ""))     # coords on the source wav
            blocks     .append((cursor, cursor + take, ""))  # coords in assembled clip
            cursor    += take
            remaining -= take
            
        
        # ----- short-utterance handling (whole utterance first) -------------
        # short_utt = noisy.shape[-1] <= tgt_N
        if short_utt:
            full_txt = self._norm_txt(" ".join(w for *_, w in iv)) if iv else ""
            spans    = [(0, noisy.shape[-1], full_txt)]   # one span = whole utt
            first    = None                               # skip large-cut step

            chosen_src.append(spans[0])
            blocks.append((cursor, cursor + noisy.shape[-1], full_txt))
            cursor    += noisy.shape[-1]
            remaining -= noisy.shape[-1]
            remaining = max(remaining, 0)     # safety


        
        # -------- first (large) cut -------------------------------------------
        if not short_utt:      # <-- skip for short utterances
            big_thr = self.big_cut_min * tgt_N
            fit     = [c for c in spans if c[1] - c[0] <= remaining]

            if fit:                                           # ← only if something fits
                big    = [c for c in fit if c[1] - c[0] >= big_thr]
                first  = rng.choice(big) if big else max(fit, key=lambda x: x[1] - x[0])

                chosen_src.append(first)
                length = first[1] - first[0]
                blocks.append((cursor, cursor + length, first[2]))
                cursor    += length
                remaining -= length
            else:
                first = None          # no speech block fits → clip will be filled with noise

    
        
        # -------- extra cuts with spacing --------------------------------------
        space_N = int(rng.uniform(self.spacing_min, self.spacing_max) * fs)
        pool = [c for c in spans if c is not first]
        pool = rng.sample(pool, len(pool)) if rng.random() < self.p_random \
            else sorted(pool, key=lambda x: x[1] - x[0], reverse=True)

        for s, e, w in pool:
            need = (e - s) + (space_N if space_N else 0)
            # if need > remaining or remaining < self.min_N:
            if need > remaining or remaining < self.min_text_N:
                continue

            # ---------- 1) spacing noise (safe) --------------------------------
            if space_N and noise_blocks:
                nb      = rng.choice(noise_blocks)          # (ns, ne)
                avail   = nb[1] - nb[0]
                take    = min(space_N, avail)               # never larger!

                # if take >= self.min_N:                      # still useful
                if take >= self.min_noise_N:                      # still useful
                    if avail == take:                       # block too short → whole
                        slice_s, slice_e = nb
                    else:                                   # long enough → random sub-slice
                        slice_s = rng.randint(nb[0], nb[1] - take)
                        slice_e = slice_s + take

                    chosen_src.append((slice_s, slice_e, ""))          # coords in source
                    blocks.append((cursor, cursor + take, ""))         # coords in clip
                    cursor    += take
                    remaining -= take

            # ---------- 2) the word span itself --------------------------------
            if e - s <= remaining:
                chosen_src.append((s, e, w))
                blocks.append((cursor, cursor + (e - s), w))
                cursor    += e - s
                remaining -= e - s
            # if remaining < self.min_N:
            if remaining < self.min_text_N:    
                break

        

        # tail noise fill
        for ns,ne in noise_blocks:
            if remaining==0: break
            take=min(ne-ns,remaining)
            chosen_src.append((ne-take,ne,""))
            blocks.append((cursor,cursor+take,""))
            cursor+=take; remaining-=take

        # pad if any
        if remaining:
            mask[-remaining:]=0.
            pad=torch.zeros_like(clean[:,:remaining])
            noisy=torch.cat([noisy, pad],-1) if noisy.shape[-1]<tgt_N else noisy
            clean=torch.cat([clean, pad],-1) if clean.shape[-1]<tgt_N else clean

        # crop/concat selected pieces from source
        wav_c=torch.cat([clean[:,s:e] for s,e,_ in chosen_src],-1)
        wav_n=torch.cat([noisy[:,s:e] for s,e,_ in chosen_src],-1)

        # txt=" ".join(w for *_,w in reindex_blocks(blocks) if w).lower().strip()
        raw = " ".join(w for *_, w in reindex_blocks(blocks) if w)
        txt = self._norm_txt(raw)   
        
        # make absolutely sure all returned tensors share the same length
        if wav_n.shape[-1] < tgt_N:
            pad = (0, tgt_N - wav_n.shape[-1])
            wav_n  = torch.nn.functional.pad(wav_n,  pad)
            wav_c  = torch.nn.functional.pad(wav_c,  pad)
            mask   = torch.nn.functional.pad(mask,   pad, value=0.0)

                
        
        # return wav_n, wav_c, txt, mask
    
    # ----------- NEW: lightweight per-sample meta ----------------
        meta = {
            "clean_fn": str(self.clean_path / fn) if self.clean_available else "",
            "noisy_fn": str(self.noisy_path / fn),
            "blocks":   [(int(s), int(e), w) for s, e, w in chosen_src],
        }

        return wav_n, wav_c, txt, mask, meta          #  ← extra field

