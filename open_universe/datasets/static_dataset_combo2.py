# static_dataset_combo.py

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
import time

# optional progress bar
try:
    from tqdm import tqdm
except ImportError:   # tqdm not installed → dummy
    tqdm = None

log = logging.getLogger(__name__)


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
        skip_no_text: bool  = True,        # ← NEW: ignore <not-available>
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

        # # filter on length
        # self.file_list, self.lengths = [], []
        # for f in files:
        #     n = torchaudio.info(str(self.noisy_path / f)).num_frames
        #     if n <= self.max_len:
        #         self.file_list.append(f)
        #         self.lengths.append(n)
        
        print(f"skip_no_text: {skip_no_text}")
        
        # length filter  (+ optional skip_no_text)
        self.file_list, self.lengths = [], []
        for f in files:
            n = torchaudio.info(str(self.noisy_path / f)).num_frames
            if n > self.max_len:
                continue

            if skip_no_text and text_path:
                txt_file = Path(to_absolute_path(text_path)) / f"{Path(f).stem}.txt"
                if not txt_file.exists() or txt_file.read_text().strip() == "<not-available>":
                    continue

            self.file_list.append(f)
            self.lengths.append(n)
            
            
        # -------- manifest cache (speeds up large corpora) ------------ #
        manifest = self.noisy_path.parent / f"{split}_manifest.pkl"
        if manifest.exists():
            self.file_list, self.lengths = torch.load(manifest)
            log.info(f"[{split}] loaded manifest → {len(self.file_list)} samples")
        else:
            from multiprocessing.pool import ThreadPool

            n_threads = min(32, os.cpu_count() or 1)
            log.info(f"[{split}] scanning {len(files)} files "
                     f"with {n_threads} threads …")
            print(f"[{split}] scanning {len(files)} files "
                  f"with {n_threads} threads …")

            def probe(f):
                n = torchaudio.info(str(self.noisy_path / f)).num_frames
                if n > self.max_len:
                    return None
                if skip_no_text and text_path:
                    txt = Path(to_absolute_path(text_path)) / f"{Path(f).stem}.txt"
                    if (not txt.exists()) or txt.read_text().strip() == "<not-available>":
                        return None
                return f, n

            # self.file_list, self.lengths = [], []
            # with ThreadPool(16) as pool:                       # <<< adjust threads
            #     for out in pool.imap_unordered(probe, files):
            #         if out is not None:
            #             f, n = out
            #             self.file_list.append(f)
            #             self.lengths.append(n)
            
            
            self.file_list, self.lengths = [], []

            bar = tqdm(total=len(files), unit="file",
                       desc=f"{split} scan") if tqdm else None
            t0  = time.time()
            print(f"starting scan")
            with ThreadPool(n_threads) as pool:
                for out in pool.imap_unordered(probe, files):
                    if out is not None:
                        f, n = out
                        self.file_list.append(f)
                        self.lengths.append(n)
                    if bar: bar.update()
            if bar: bar.close()

            log.info(f"[{split}] scanned in {time.time()-t0:.1f}s")

            torch.save((self.file_list, self.lengths), manifest)
            log.info(f"[{split}] wrote manifest {manifest}")    
            print(f"[{split}] wrote manifest {manifest}")    
            
        

        self.text_path = Path(to_absolute_path(text_path)) if text_path else None
        log.info(
            f"[{split}] {len(self.file_list)} files (≤ {max_len_sec}s) "
            f"fixed_len={fixed_len_sec}"
        )

    # ------------------------------------------------------------------ #
    def __len__(self): return len(self.file_list)

    def _load(self, p: Path) -> torch.Tensor:
        # wav, sr = torchaudio.load(p)
        wav, sr = torchaudio.load(p, backend="soundfile")
        if sr != self.fs:
            wav = torchaudio.functional.resample(wav, sr, self.fs)
        return wav

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        fn = self.file_list[idx]
        noisy = self._load(self.noisy_path / fn)
        clean = (
            self._load(self.clean_path / fn)
            if self.clean_available
            else torch.zeros_like(noisy)
        )

        # transcript
        if self.text_path and (p := self.text_path / f"{Path(fn).stem}.txt").exists():
            txt = p.read_text().strip()
        else:
            txt = ""

        T = noisy.shape[-1]
        if self.split == "train" and self.fixed_len:
            if T < self.fixed_len:
                pad = self.fixed_len - T
                noisy = torch.nn.functional.pad(noisy, (0, pad))
                clean = torch.nn.functional.pad(clean, (0, pad))
                mask  = torch.cat([torch.ones(T), torch.zeros(pad)]).float()
            else:                           # equal (longer clips were filtered)
                mask = torch.ones(self.fixed_len)
        else:                               # val / test
            mask = torch.ones(T)

        return noisy, clean, txt, mask
