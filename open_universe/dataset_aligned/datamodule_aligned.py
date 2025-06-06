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
import torchaudio, os
from pathlib import Path


def _pad_to_len(t: torch.Tensor, L: int) -> torch.Tensor:
    """1-D right-pad (works for [C,T] or [T])"""
    return torch.nn.functional.pad(t, (0, L - t.shape[-1]))


def max_collator(batch):
    """Pad to the longest sample and stack."""
    noisy, clean, txt, mask = zip(*batch)
    max_len = max(x.shape[-1] for x in noisy)
    pad_t   = lambda t: torch.nn.functional.pad(t, (0, max_len-t.shape[-1]))
    noisy = torch.stack([pad_t(x) for x in noisy])
    clean = torch.stack([pad_t(x) for x in clean])
    mask  = torch.stack([pad_t(m) for m in mask])
    
    return noisy, clean, list(txt), mask


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

    def __init__(self, train, val, test, datasets,
                 train_sample_folder: str | None = None,
                 train_sample_batches: int = 0,
                 val_sample_folder: str | None = None,
                 val_sample_batches: int = 0):
        super().__init__()
        self.cfg = dict(train=train, val=val, test=test)
        self.datasets_list = datasets
        self.datasets = {}
        
        
        # ---------- new debug-dump settings -------------------------
        self._dump_cfg = {
            "train": dict(dir=Path(train_sample_folder) if train_sample_folder else None,
                          max=max(0, train_sample_batches), cnt=0),
            "val":   dict(dir=Path(val_sample_folder)   if val_sample_folder   else None,
                          max=max(0, val_sample_batches),   cnt=0),
        }
        for s, d in self._dump_cfg.items():
            if d["dir"]:
                print(f"Debug dump for {s}: {d['dir']}")
                for sub in ("clean", "noisy", "txt"):
                    (d["dir"]/sub).mkdir(parents=True, exist_ok=True)
        
        
        
     # ------------------------------------------------------------------ #
    # def _make_collator(self):
    #     """Return a closure so it can access self.*"""
    #     def collate(batch):
    #         noisy, clean, txt, mask = zip(*batch)
    #         L = max(x.shape[-1] for x in noisy)

    #         noisy = torch.stack([_pad_to_len(x, L) for x in noisy])
    #         clean = torch.stack([_pad_to_len(x, L) for x in clean])
    #         mask  = torch.stack([_pad_to_len(m, L) for m in mask])

    #         # ------ optional one-time export of the very first batches ------
    #         if self._dump_dir and self._dump_cnt < self._dump_max:   # <- use the _-prefixed vars
    #             sr = 16_000
    #             B  = noisy.shape[0]
    #             for i in range(B):
    #                 uid = f"b{self._dump_cnt:02d}_{i:02d}"
    #                 torchaudio.save(self._dump_dir/'noisy'/f"{uid}.wav",
    #                                 noisy[i].cpu(), sr)
    #                 torchaudio.save(self._dump_dir/'clean'/f"{uid}.wav",
    #                                 clean[i].cpu(), sr)
    #                 (self._dump_dir/'txt'/f"{uid}.txt").write_text(txt[i])
    #             self._dump_cnt += 1                                # <- update the counter

    #         return noisy, clean, list(txt), mask
    #     return collate
    
    
    def _make_collator(self, split: str):
        dump = self._dump_cfg[split]

        def collate(batch):
            noisy, clean, txt, mask = zip(*batch)
            L = max(x.shape[-1] for x in noisy)

            pad = lambda t: torch.nn.functional.pad(t, (0, L - t.shape[-1]))
            noisy = torch.stack([pad(x) for x in noisy])
            clean = torch.stack([pad(x) for x in clean])
            mask  = torch.stack([pad(m) for m in mask])

            # -------- optional on-the-fly export -----------------
            if dump["dir"] and dump["cnt"] < dump["max"]:
                sr = 16_000
                for i in range(noisy.shape[0]):
                    uid = f"b{dump['cnt']:02d}_{i:02d}"
                    torchaudio.save(dump["dir"] / "noisy" / f"{uid}.wav", noisy[i].cpu(), sr, backend="soundfile")
                    torchaudio.save(dump["dir"] / "clean" / f"{uid}.wav", clean[i].cpu(), sr, backend="soundfile")
                    (dump["dir"] / "txt" / f"{uid}.txt").write_text(txt[i])
                dump["cnt"] += 1
            return noisy, clean, list(txt), mask

        return collate

    

    def setup(self, *args, **kwargs):
        for split in ["train", "val", "test"]:
            ds_name = self.cfg[split].dataset
            self.datasets[split] = instantiate(
                self.datasets_list[ds_name], _recursive_=False
            )

    def _get_dataloader(self, split):
        return torch.utils.data.DataLoader(
            self.datasets[split],
            collate_fn=self._make_collator(split),   # ← here we supply "train", "val", or "test"
            **self.cfg[split].dl_opts,
        )

        
    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")
