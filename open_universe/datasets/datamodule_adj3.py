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
A generic pytorch-lightning DataModule object configurble with Hydra

Author: Robin Scheibler (@fakufaku)
"""
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate

from torch.nn.utils.rnn import pad_sequence           # ← handy utility
from typing import List, Tuple

from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
import math
import random


def max_collator(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
):
    """
    Args
    ----
    batch : list of samples where each sample is
            (noisy [C,T_i], clean [C,T_i], transcript str)

    Returns
    -------
    noisy_b : Tensor [B, C, T_max]
    clean_b : Tensor [B, C, T_max]
    texts   : list[str]
    mask_b  : Tensor [B, T_max]   (1 for real audio, 0 for padding)
    """

    noisy_list, clean_list, texts = zip(*batch)

    # ------------------------------------------------------------
    # 1) Transpose to [T_i, C] so pad_sequence can work
    # ------------------------------------------------------------
    noisy_tc = [n.transpose(0, 1) for n in noisy_list]   # → [T_i, C]
    clean_tc = [c.transpose(0, 1) for c in clean_list]

    # ------------------------------------------------------------
    # 2) Pad to longest sequence in this batch
    #    Result: [B, T_max, C]
    # ------------------------------------------------------------
    noisy_padded = pad_sequence(noisy_tc, batch_first=True, padding_value=0.0)
    clean_padded = pad_sequence(clean_tc, batch_first=True, padding_value=0.0)

    # ------------------------------------------------------------
    # 3) Permute back to [B, C, T_max]
    # ------------------------------------------------------------
    noisy_b = noisy_padded.permute(0, 2, 1).contiguous()
    clean_b = clean_padded.permute(0, 2, 1).contiguous()

    # ------------------------------------------------------------
    # 4) Build valid‑frame mask
    # ------------------------------------------------------------
    lengths = torch.tensor([w.shape[-1] for w in noisy_list], dtype=torch.long)
    T_max = noisy_b.shape[-1]
    mask_b = (torch.arange(T_max, device=lengths.device)[None, :] <
              lengths[:, None]).float()
    
    # ----------------- NEW: stats ---------------------------------
    # total valid (non‑padded) frames across the whole batch
    valid = mask_b.sum().item()
    total  = mask_b.numel()
    pad_pct = 100 * (1 - valid / total)
    # print once every N batches to avoid spamming
    # if random.random() < 0.05:        # 5 % of the time
    
    B      = noisy_b.shape[0]                 # clips in this batch
    valid  = mask_b.sum(dim=1)                # frames per‑clip
    pad_pc = 100 * (1 - valid / T_max)        # padding % per‑clip
    print(f"[collate] B={B:2d}  max_len={T_max}  "
          f"avg_pad={pad_pc.mean():4.1f}%  "
          f"max_pad={pad_pc.max():4.1f}%  "
          f"tot_pad={pad_pct:4.1f}%"
          )

    return noisy_b, clean_b, list(texts), mask_b


from inspect import signature
_ALLOWED = set(signature(torch.utils.data.DataLoader).parameters)

def _filtered(opts):
    "drop any Hydra option that DataLoader doesn’t understand"
    return {k: v for k, v in opts.items() if k in _ALLOWED}


class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that can be used to train, validate and test

    Parameters
    ----------
    train : OmegaConf
        Configuration for the training dataset
    val : OmegaConf
        Configuration for the validation dataset
    test : OmegaConf
        Configuration for the test dataset
    datasets : Dict[str, Any]
        A dictionary with the config of all possible datasets
    """

    def __init__(self, train, val, test, datasets):
        super().__init__()
        self.cfg = dict(train=train, val=val, test=test)
        self.datasets_list = datasets
        self.datasets = {}

    def setup(self, *args, **kwargs):
        for split in ["train", "val", "test"]:
            ds_name = self.cfg[split].dataset
            self.datasets[split] = instantiate(
                self.datasets_list[ds_name], _recursive_=False
            )

    def _get_dataloader(self, split):
        ds = self.datasets[split]

        if split == "train":
            budget_frames = int(ds.audio_batch_size * ds.fs)
            sampler = VariableBatchSampler(
                ds.lengths,
                budget_frames=budget_frames,
                fs=ds.fs,
                shuffle=True,
            )
            dl_opts = _filtered(self.cfg[split].dl_opts)
            # make sure we don’t pass batch_size / shuffle alongside batch_sampler
            dl_opts.pop("batch_size", None)
            dl_opts.pop("shuffle",    None)
            return DataLoader(
                ds,
                batch_sampler=sampler,
                collate_fn=max_collator,
                **dl_opts,
            )

        # ---------------- val / test -----------------
        dl_opts = _filtered(self.cfg[split].dl_opts)    # <- removes audio_batch_size
        return DataLoader(
            ds,
            collate_fn=max_collator,
            **dl_opts,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")


import random, math
from torch.utils.data import BatchSampler

class VariableBatchSampler(BatchSampler):
    """
    Packs clips into mini‑batches whose *total* length ≤ `budget_frames`.
    Buckets clips that are within ±`width_sec` of each other so padding stays low.
    The buckets themselves are NOT shuffled, so training starts with short clips.
    """

    def __init__(
        self,
        lengths,
        budget_frames,
        fs,
        width_sec=0.25, # 0.5
        shuffle=True,            # shuffle **inside** each bucket
        sampler=None,            # Lightning may inject DistributedSampler
        **_ignored,
    ):
        self.shuffle        = shuffle
        self.budget_frames  = budget_frames
        self.lengths        = lengths          # keep for __iter__ / __len__
        width_frames        = int(width_sec * fs)

        # ---- bucket clips by similar length --------------------------------
        ids = list(sampler) if sampler is not None else list(range(len(lengths)))
        ids = sorted(ids, key=lengths.__getitem__)            # ascending length
        self.buckets, cur = [], [ids[0]]
        for i in ids[1:]:
            if abs(lengths[i] - lengths[cur[0]]) > width_frames:
                self.buckets.append(cur); cur = [i]
            else:
                cur.append(i)
        self.buckets.append(cur)

    # --------------------------------------------------------------------- #
    def __iter__(self):
        for bucket in self.buckets:                       # shortest → longest
            if self.shuffle:
                random.shuffle(bucket)                    # permute inside
            batch, tot = [], 0
            for idx in bucket:
                l = self.lengths[idx]
                if tot + l > self.budget_frames and batch:
                    yield batch
                    batch, tot = [], 0
                batch.append(idx)
                tot += l
            if batch:
                yield batch

    # --------------------------------------------------------------------- #
    def __len__(self):
        """Exact number of batches yielded by __iter__ (deterministic)."""
        n_batches = 0
        for bucket in self.buckets:
            tot = 0
            for idx in bucket:
                l = self.lengths[idx]
                if tot + l > self.budget_frames and tot > 0:
                    n_batches += 1
                    tot = 0
                tot += l
            if tot > 0:
                n_batches += 1
        return n_batches
    

# import random, math
# from torch.utils.data import BatchSampler
# from typing import List

# class VariableBatchSampler(BatchSampler):
#     """
#     Packs clips into mini‑batches whose *total* length ≤ `budget_frames`.
#     Clips are bucketed by **relative duration**: every clip in a bucket is at
#     most `width_pct` longer than the shortest clip in that bucket.

#     Example
#     -------
#     >>> sampler = VariableBatchSampler(lengths, budget_frames=512000, width_pct=0.05)
#     # clips inside a bucket differ ≤ 5 % in length
#     """

#     def __init__(
#         self,
#         lengths: List[int],
#         budget_frames: int,
#         width_pct: float = 0.05,   # 5 % tolerance (default)
#         shuffle: bool = True,
#         sampler=None,              # Lightning may pass DistributedSampler
#         **_ignored,
#     ):
#         self.lengths        = lengths
#         self.budget_frames  = budget_frames
#         self.shuffle        = shuffle
#         self.width_pct      = width_pct

#         # -------- sort ids by length ---------------------------------------
#         ids = list(sampler) if sampler is not None else list(range(len(lengths)))
#         ids = sorted(ids, key=lengths.__getitem__)

#         # -------- build relative‑width buckets -----------------------------
#         self.buckets, cur = [], [ids[0]]
#         for i in ids[1:]:
#             shortest = lengths[cur[0]]
#             if lengths[i] - shortest > self.width_pct * shortest:
#                 self.buckets.append(cur)
#                 cur = [i]
#             else:
#                 cur.append(i)
#         self.buckets.append(cur)

#     # --------------------------------------------------------------------- #
#     def __iter__(self):
#         if self.shuffle:
#             random.shuffle(self.buckets)          # keeps short→long optional
#         for bucket in self.buckets:
#             if self.shuffle:
#                 random.shuffle(bucket)            # permute inside bucket
#             batch, tot = [], 0
#             for idx in bucket:
#                 l = self.lengths[idx]
#                 if tot + l > self.budget_frames and batch:
#                     yield batch
#                     batch, tot = [], 0
#                 batch.append(idx)
#                 tot += l
#             if batch:
#                 yield batch

#     # --------------------------------------------------------------------- #
#     def __len__(self):
#         n_batches = 0
#         for bucket in self.buckets:
#             tot = 0
#             for idx in bucket:
#                 l = self.lengths[idx]
#                 if tot + l > self.budget_frames and tot > 0:
#                     n_batches += 1
#                     tot = 0
#                 tot += l
#             if tot > 0:
#                 n_batches += 1
#         return n_batches


