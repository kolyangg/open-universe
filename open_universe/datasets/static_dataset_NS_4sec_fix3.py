import logging
import math
import os
import random
from pathlib import Path
from typing import Optional, Union, List, Tuple
# from torchaudio.backend import sox_io_backend   # faster length query

import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class NoisyDataset(Dataset):
    """
    A dataset that loads noisy / clean waveform pairs (and optional transcripts)
    and leaves *all* padding & cropping to its collate function.

    Use:
        ds = NoisyDataset(...)
        loader = DataLoader(
            ds,
            batch_size=16,
            shuffle=True,
            collate_fn=ds.get_collate_fn(  # <- crucial
                random_crop=True          # crop clips that exceed `audio_len`
            ),
        )
    """
    # --------------------------------------------------------------------- #
    # INITIALISATION
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        audio_path: Union[str, Path],
        # audio_len: Union[int, float] = 4,      # seconds (acts as *max* length)
        audio_len: Union[int, float] =  8.0, #  16.0,   # hard cut‑off (filter only)
        audio_batch_size: Union[int, float] =  16.0, # 8.0 = 48 min, # 10.0 = 41min, 12.0 = 37min, # 14.0 = 54min # 12.0, # 8.0, # 16.0, # 32.0,  # NEW – per‑batch budget (s)
        fs: Optional[int] = 16000,
        split: Optional[str] = "train",
        noisy_folder: Optional[str] = "noisy",
        clean_folder: Optional[str] = "clean",
        text_path: Optional[str] = None,
        part_used: Optional[float] = 1.0,
    ):
        super().__init__()

        audio_path = Path(to_absolute_path(str(audio_path)))
        if split is not None:
            audio_path = audio_path / split

        self.noisy_path = audio_path / noisy_folder

        if clean_folder is None:
            self.clean_available = False
        else:
            self.clean_path = audio_path / clean_folder
            self.clean_available = self.clean_path.exists()

        if not self.noisy_path.exists():
            raise FileNotFoundError(f"{self.noisy_path} does not exist")

        # ------------------------------------------------------------------ #
        # Build the file list
        # ------------------------------------------------------------------ #
        self.file_list = sorted(os.listdir(self.noisy_path))
        if self.clean_available:
            common = set(self.file_list) & set(os.listdir(self.clean_path))
            self.file_list = sorted(common)

        # Optionally use only a part of the dataset
        if part_used < 1.0:
            n_keep = max(1, int(len(self.file_list) * part_used))
            self.file_list = self.file_list[:n_keep]
            log.info(f"Using {part_used:.2%} of dataset -> {len(self.file_list)} files")

        # Convert audio_len (seconds) to samples for later cropping
        self.audio_len = int(audio_len * fs) if audio_len is not None else None
        self.fs = fs
        self.audio_batch_size = audio_batch_size   # ← expose budget for sampler
        self.split = split

        # Optional transcripts
        self.text_path = Path(to_absolute_path(text_path)) if text_path else None
        

        # ------------------------------------------------------------------ #
        # Filter out files longer than `audio_len` *during training only*.
        # (We keep them in test/val so we can report on full‑length clips.)
        # ------------------------------------------------------------------ #
        # 1) filter
        if self.audio_len and split in {"train", "val"}:
            keep = [fn for fn in self.file_list
                    if torchaudio.info(str(self.noisy_path/fn)).num_frames <= self.audio_len]
            self.file_list = keep
        # 2) cache lengths AFTER filtering
        self.lengths = [torchaudio.info(str(self.noisy_path/fn)).num_frames
                  for fn in self.file_list]

    # --------------------------------------------------------------------- #
    # BASIC DATASET API
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.file_list)

    def _load_wave(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.fs:
            wav = torchaudio.functional.resample(wav, sr, self.fs)
        return wav  # shape: [C, T]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return *raw*, variable‑length tensors – no padding!"""
        key = self.file_list[idx]

        noisy = self._load_wave(self.noisy_path / key)

        if self.clean_available:
            clean = self._load_wave(self.clean_path / key)
        else:
            clean = torch.zeros_like(noisy)

        transcript = ""
        if self.text_path:
            tfile = self.text_path / (Path(key).stem + ".txt")
            if tfile.exists():
                transcript = tfile.read_text().strip()

        return noisy, clean, transcript