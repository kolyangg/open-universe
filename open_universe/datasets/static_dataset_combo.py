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
