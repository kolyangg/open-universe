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


def max_collator(batch):
    """
    Collate a bunch of multichannel signals based
    on the size of the longest sample. The samples are cut at the center
    """
    max_len = max([s[0].shape[-1] for s in batch])

    new_batch = []
    for bidx, row in enumerate(batch):
        new_row = []
        for eidx, el in enumerate(row):
            if isinstance(el, torch.Tensor):
                off = max_len - el.shape[-1]
                new_row.append(torch.nn.functional.pad(el, (0, off)))
            else:
                new_row.append(el)
        new_batch.append(tuple(new_row))

    # return torch.utils.data.default_collate(new_batch)

    batch_out = torch.utils.data.default_collate(new_batch)
    # ------------ quick padding diagnostics ------------------------------
    noisy_b, _, _, mask_b = batch_out            # unpack tensors
    T = noisy_b.shape[-1]
    pad_pc = 100 * (1 - mask_b.sum(dim=1) / T)   # % padding per clip
    print(f"[collate] B={noisy_b.size(0):2d}  "
        f"max_len={T}  avg_pad={pad_pc.mean():4.1f}%  "
        f"max_pad={pad_pc.max():4.1f}%")
    
    # -------- NEW: send the same numbers to W&B ------------------
    try:                         # only if wandb is available & a run is active
        import wandb
        if wandb.run is not None:
            wandb.run.log(
                {
                    "batch_checks/avg_pad": pad_pc.mean().item(),
                    "batch_checks/max_pad": pad_pc.max().item(),
                    "batch_checks/B":        noisy_b.size(0),
                    "batch_checks/max_len":  T,
                },
                commit=False,    # keep the Lightning step syncing behaviour
            )
    except ImportError:
        pass

        
    return batch_out


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
        return torch.utils.data.DataLoader(
            self.datasets[split],
            collate_fn=max_collator,
            **self.cfg[split].dl_opts,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")
