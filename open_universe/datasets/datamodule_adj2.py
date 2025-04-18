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
    print(f"[collate] max_len={T_max} samples  |  padding={pad_pct:.1f}%")

    return noisy_b, clean_b, list(texts), mask_b


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

        # Use bucketed sampler *only* for training
        if split == "train":
            bs = self.cfg[split].dl_opts.batch_size
            sampler = BucketBatchSampler(ds.lengths, batch_size=bs, fs=ds.fs, shuffle=True)
            return DataLoader(
                ds,
                batch_sampler=sampler,
                collate_fn=max_collator,
                num_workers=self.cfg[split].dl_opts.num_workers,
                pin_memory=True,
            )

        # val / test remain unchanged
        return DataLoader(
            ds,
            collate_fn=max_collator,
            **self.cfg[split].dl_opts,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")





class BucketBatchSampler(BatchSampler):
    """
    Groups indices with similar lengths so that each mini‑batch contains
    clips whose duration differs by no more than `bucket_width` samples.
    """

    
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        fs: int,
        width_sec: float = 0.5,
        shuffle: bool = True,
        sampler=None,        # <‑‑‑ NEW, Lightning passes this
        **_ignored,          # swallow any other injected kw
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        max_diff = int(width_sec * fs)

        # If Lightning gives us a sampler, use its ordering; else range(len)
        ids = list(sampler) if sampler is not None else list(range(len(lengths)))
        ids = sorted(ids, key=lambda i: lengths[i])

        self.buckets, cur = [], [ids[0]]
        for i in ids[1:]:
            if lengths[i] - lengths[cur[0]] > max_diff:
                self.buckets.append(cur); cur = [i]
            else:
                cur.append(i)
        self.buckets.append(cur)
    
    def __iter__(self):
        if self.shuffle:
            import random; random.shuffle(self.buckets)
        for bucket in self.buckets:
            # inside each bucket we can still random‑sample mini‑batches
            if self.shuffle: random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i+self.batch_size]
    def __len__(self):
        return sum(math.ceil(len(b)/self.batch_size) for b in self.buckets)