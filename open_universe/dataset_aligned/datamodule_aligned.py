# SPDX‑License‑Identifier: Apache‑2.0
"""Lightning DataModule wrapper for WordCutDataset."""
from __future__ import annotations
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra.utils import instantiate

import torch
import logging
log = logging.getLogger(__name__)

class WordCutDataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, datasets, **hp):
        super().__init__()
        self.split_cfg = dict(train=train, val=val, test=test)
        self.ds_cfgs, self.h = datasets, hp
        
        # same diagnostic switches as combo version
        self.do_print  = hp.get("print_collate_log",  True)
        self.do_wandb  = hp.get("wandb_collate_log",  True)

        
        # --- fixed‑length → trivial collate: stack tensors -------------
        def wordcut_collate(batch, *,
                            do_print=self.do_print,
                            do_wandb=self.do_wandb):
            """
            Stack noisy/clean/mask tensors; pad to the longest sample
            (normally they are equal already).  Text is a plain list.
            """
            noisy, clean, txt, mask = zip(*batch)
            max_len = max(x.shape[-1] for x in noisy)

            n_pad, c_pad, m_pad = [], [], []
            for n, c, m in zip(noisy, clean, mask):
                pad = max_len - n.shape[-1]
                if pad:
                    n = torch.nn.functional.pad(n, (0, pad))
                    c = torch.nn.functional.pad(c, (0, pad))
                    m = torch.nn.functional.pad(m, (0, pad), value=0.0)
                n_pad.append(n); c_pad.append(c); m_pad.append(m)

            noisy_b = torch.stack(n_pad)
            clean_b = torch.stack(c_pad)
            mask_b  = torch.stack(m_pad)

            # ----------- diagnostics (matches datamodule_combo) --------
            if do_print or do_wandb:
                T       = noisy_b.shape[-1]
                pad_pc  = 100 * (1 - mask_b.sum(dim=1) / T)
                avg_pad = pad_pc.mean().item()
                max_pad = pad_pc.max().item()
                if do_print:
                    print(f"[collate] B={noisy_b.size(0):2d}  max_len={T}  "
                          f"avg_pad={avg_pad:4.1f}%  max_pad={max_pad:4.1f}%")

                if do_wandb:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.run.log(
                                {"batch_checks/avg_pad": avg_pad,
                                 "batch_checks/max_pad": max_pad,
                                 "batch_checks/B": noisy_b.size(0),
                                 "batch_checks/max_len": T},
                                commit=False,
                            )
                    except ImportError:
                        pass

            return noisy_b, clean_b, list(txt), mask_b
            
    
        self.collate_fn = wordcut_collate

    def setup(self, stage=None):
        self.datasets = {s: instantiate(self.ds_cfgs[c.dataset], _recursive_=False)
                         for s,c in self.split_cfg.items()}

    def train_dataloader(self):
        cfg = self.split_cfg["train"].dl_opts
        return DataLoader(self.datasets["train"],
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        cfg = self.split_cfg["val"].dl_opts
        return DataLoader(self.datasets["val"],
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          num_workers=cfg.num_workers)

    def test_dataloader(self):
        cfg = self.split_cfg["test"].dl_opts
        return DataLoader(self.datasets["test"],
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          collate_fn=self.collate_fn,
                          num_workers=cfg.num_workers)
