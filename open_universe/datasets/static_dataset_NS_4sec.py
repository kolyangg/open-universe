# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# NoisyDataset module

A simple dataset for speech enhancement where the clean/noisy samples are
stored in two different folders.

Author: Robin Scheibler (@fakufaku)
"""
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class NoisyDataset(Dataset):
    def __init__(
        self,
        audio_path: Union[str, Path],
        audio_len: Union[int, float] = 4,
        fs: Optional[int] = 16000,
        split: Optional[str] = "train",
        noisy_folder: Optional[str] = "noisy",
        clean_folder: Optional[str] = "clean",
        text_path: Optional[str] = None,
        part_used: Optional[float] = 1.0,
    ):
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

        # Build the file list.
        self.file_list = sorted(os.listdir(self.noisy_path))
        if self.clean_available:
            common_files = set(self.file_list) & set(os.listdir(self.clean_path))
            self.file_list = sorted(list(common_files))
        
        if part_used < 1.0:
            n_files = len(self.file_list)
            n_keep = max(1, int(n_files * part_used))
            self.file_list = self.file_list[:n_keep]
            log.info(f"Using {part_used:.2%} of dataset: {len(self.file_list)} files")

        # Convert audio_len (seconds) to samples.
        if audio_len is not None:
            self.audio_len = int(audio_len * fs)
        else:
            self.audio_len = None
        self.fs = fs
        self.split = split

        # NEW: store text path if provided.
        self.text_path = Path(to_absolute_path(text_path)) if text_path else None

        # --- NEW SECTION: Filter out files longer than 4 seconds ---
        if self.audio_len is not None:
            filtered_files = []
            for f in self.file_list:
                filepath = self.noisy_path / f
                info = torchaudio.info(str(filepath))
                if info.num_frames <= self.audio_len:
                    filtered_files.append(f)
            self.file_list = filtered_files
            log.info(f"Filtered files: {len(self.file_list)} files below {audio_len} sec.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        noisy_path = self.noisy_path / self.file_list[idx]
        key = noisy_path.stem

        noisy, sr = torchaudio.load(noisy_path)
        if self.clean_available:
            clean_path = self.clean_path / self.file_list[idx]
            clean, sr2 = torchaudio.load(clean_path)
            assert sr == sr2
        else:
            clean = 0

        if self.text_path is not None:
            text_file = self.text_path / f"{key}.txt"
            if text_file.exists():
                with open(text_file, "r") as f:
                    transcript = f.read().strip()
            else:
                transcript = ""
        else:
            transcript = ""

        if self.split == "test":
            return noisy, clean, transcript, torch.empty(0) # dummy mask for test (TO CHECK)

        # Updated padding: pad with zeros and create a valid-frame mask.
        if self.audio_len is not None:
            ori_len = noisy.shape[-1]
            if ori_len < self.audio_len:
                pad_amount = self.audio_len - ori_len
                noisy = torch.nn.functional.pad(noisy, (0, pad_amount))
                if self.clean_available:
                    clean = torch.nn.functional.pad(clean, (0, pad_amount))
                # Create mask: 1 for real frames, 0 for padded frames.
                valid_mask = torch.cat([torch.ones(ori_len), torch.zeros(pad_amount)])
            else:
                st_idx = random.randint(0, ori_len - self.audio_len)
                noisy = noisy[..., st_idx: st_idx + self.audio_len]
                if self.clean_available:
                    clean = clean[..., st_idx: st_idx + self.audio_len]
                valid_mask = torch.ones(self.audio_len)
        else:
            valid_mask = None

        return noisy, clean, transcript, valid_mask
