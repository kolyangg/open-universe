# SPDX‑License‑Identifier: Apache‑2.0
"""DataModule with 3 batching modes + optional collate‑time logging."""
from __future__ import annotations
import math, random, logging
from typing import List, Tuple, Optional
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader, BatchSampler
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
#                         COLLATOR FACTORY                              #
# --------------------------------------------------------------------- #
def make_collator(do_print: bool, do_wandb: bool):
    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]]):
        noisy, clean, txt, mask = zip(*batch)
        noisy_tc = [x.transpose(0, 1) for x in noisy]
        clean_tc = [x.transpose(0, 1) for x in clean]
        noisy_pad = pad_sequence(noisy_tc, batch_first=True)
        clean_pad = pad_sequence(clean_tc, batch_first=True)
        mask_pad  = pad_sequence(mask,     batch_first=True)

        noisy_b = noisy_pad.permute(0, 2, 1).contiguous()
        clean_b = clean_pad.permute(0, 2, 1).contiguous()

        # ------------- diagnostics --------------------------------------
        if do_print or do_wandb:
            T       = noisy_b.shape[-1]
            pad_pc  = 100 * (1 - mask_pad.sum(dim=1) / T)
            avg_pad = pad_pc.mean().item()
            max_pad = pad_pc.max().item()
            if do_print:
                print(f"[collate] B={len(batch):2d}  max_len={T}  "
                      f"avg_pad={avg_pad:4.1f}%  max_pad={max_pad:4.1f}%")

            if do_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.run.log(
                            {
                                "batch_checks/avg_pad": avg_pad,
                                "batch_checks/max_pad": max_pad,
                                "batch_checks/B":       len(batch),
                                "batch_checks/max_len": T,
                            },
                            commit=False,
                        )
                except ImportError:
                    pass

        return noisy_b, clean_b, list(txt), mask_pad

    return _collate


# ---------------- prim‑er: bucket samplers (unchanged) ---------------- #
class _BucketSampler(BatchSampler):
    def __init__(self, lengths, fs, width_sec, width_pct, order, sampler = None):
        self.lengths = lengths
        self.width_frames = int(width_sec * fs) if width_sec else None
        # ids = sorted(range(len(lengths)), key=lengths.__getitem__)
        
        # # use Lightning‑provided sampler order when distributed
        # ids = list(sampler) if sampler is not None \
        #       else sorted(range(len(lengths)), key=lengths.__getitem__)

        # if order == "desc": ids.reverse()
        # elif order == "rand": random.shuffle(ids)
        # self.ids = ids
        
        # keep the DistributedSampler shard *but* order it by length
        ids = list(sampler) if sampler is not None else list(range(len(lengths)))
        ids.sort(key=lengths.__getitem__)          # ①  sort ascending
        if   order == "desc": ids.reverse()        # ②  optional desc
        elif order == "rand": random.shuffle(ids)  # ③  or random
        self.ids = ids                             # ④  save
        
        self.width_pct = width_pct

    def _same_bucket(self, a, b):
        la, lb = self.lengths[a], self.lengths[b]
        if self.width_frames is not None:
            return abs(la - lb) <= self.width_frames
        return abs(la - lb) <= self.width_pct * la


class BucketBatchSampler(_BucketSampler):
    def __init__(self, lengths, batch_size, sampler = None, **kw):
        # super().__init__(lengths, **kw)
        super().__init__(lengths, sampler=sampler, **kw)   # ← keep distributed sam
        self.bs = batch_size
        self.buckets, cur = [], [self.ids[0]]
        for i in self.ids[1:]:
            if not self._same_bucket(i, cur[0]):
                self.buckets.append(cur); cur = [i]
            else:
                cur.append(i)
        self.buckets.append(cur)
        
        
    def __iter__(self):
        for b in self.buckets:
            random.shuffle(b)
            for i in range(0, len(b), self.bs):
                yield b[i:i+self.bs]

    def __len__(self):
        return sum(math.ceil(len(b)/self.bs) for b in self.buckets)


class VariableBatchSampler(_BucketSampler):
    def __init__(self, lengths, budget_frames, sampler = None, **kw):
        # super().__init__(lengths, **kw)
        super().__init__(lengths, sampler=sampler, **kw)   # ← keep distributed sampler
        self.budget = budget_frames
        self.buckets, cur = [], [self.ids[0]]
        for i in self.ids[1:]:
            if not self._same_bucket(i, cur[0]):
                self.buckets.append(cur); cur = [i]
            else:
                cur.append(i)
        self.buckets.append(cur)

    def __iter__(self):
        for b in self.buckets:
            random.shuffle(b)
            batch, tot = [], 0
            for idx in b:
                L = self.lengths[idx]
                if tot + L > self.budget and batch:
                    yield batch; batch, tot = [], 0
                batch.append(idx); tot += L
            if batch: yield batch

    def __len__(self):
        return sum(
            math.ceil(sum(self.lengths[i] for i in b) / self.budget)
            for b in self.buckets
        )


# --------------------------------------------------------------------- #
#                             DATAMODULE                                #
# --------------------------------------------------------------------- #
class DataModule(pl.LightningDataModule):
    def __init__(self, train, val, test, datasets, **hp):
        super().__init__()
        self.split_cfg = dict(train=train, val=val, test=test)
        self.ds_cfgs   = datasets
        self.h         = hp
        self.mode      = hp.get("mode", "fixed")
        # new logging switches
        self.do_print  = hp.get("print_collate_log",  True)
        self.do_wandb  = hp.get("wandb_collate_log",  True)
        self.collate_fn = make_collator(self.do_print, self.do_wandb)

    # --------------------------- datasets ------------------------------- #
    def setup(self, stage=None):
        self.datasets = {}
        for split in ["train", "val", "test"]:
            name = self.split_cfg[split].dataset
            cfg  = {**self.ds_cfgs[name]}
            if split == "train" \
               and self.mode == "fixed" \
               and "fixed_len_sec" in self.h:
                cfg["fixed_len_sec"] = self.h["fixed_len_sec"]
            self.datasets[split] = instantiate(cfg, _recursive_=False)

    # ---------------------- train loader switch ------------------------- #
    def _train_loader(self):
        ds, h = self.datasets["train"], self.h
        if self.mode == "fixed":
            opt = dict(self.split_cfg["train"].dl_opts)
            opt.pop("shuffle", None)               # avoid duplicate
            batch_sz = opt.pop("batch_size", self.h["batch_size"])
            return DataLoader(
                ds,
                batch_size=batch_sz,
                shuffle=True, # let Lightning inject DistributedSampler
                collate_fn=self.collate_fn,
                **opt,
            )
        if self.mode == "bucket_bs":
            sampler = BucketBatchSampler(
                ds.lengths,
                batch_size=h["batch_size"],
                fs=ds.fs,
                width_sec=h.get("width_sec"),
                width_pct=h.get("width_pct", 0.05),
                order=h.get("order", "asc"),
            )
            return DataLoader(ds, batch_sampler=sampler, collate_fn=self.collate_fn)
        if self.mode == "bucket_budget":
            sampler = VariableBatchSampler(
                ds.lengths,
                budget_frames=int(h["budget_sec"] * ds.fs),
                fs=ds.fs,
                width_sec=h.get("width_sec"),
                width_pct=h.get("width_pct", 0.05),
                order=h.get("order", "asc"),
            )
            return DataLoader(ds, batch_sampler=sampler, collate_fn=self.collate_fn)
        raise ValueError(f"Unknown mode {self.mode}")

    # ------------------------ public API -------------------------------- #
    def train_dataloader(self): return self._train_loader()
    def val_dataloader  (self):
        return DataLoader(
            self.datasets["val"],
            collate_fn=self.collate_fn,
            **self.split_cfg["val"].dl_opts,
        )
    def test_dataloader (self):
        return DataLoader(
            self.datasets["test"],
            collate_fn=self.collate_fn,
            **self.split_cfg["test"].dl_opts,
        )



# # SPDX‑License‑Identifier: Apache‑2.0
# """
# One DataModule, three batching modes.

# Extra keys understood at the *root* level of the YAML config:
#     mode:           fixed | bucket_bs | bucket_budget
#     fixed_len_sec:  float   (for mode=fixed)
#     batch_size:     int     (for fixed / bucket_bs)
#     budget_sec:     float   (for bucket_budget)
#     width_sec:      float   (bucket tolerance – seconds)
#     width_pct:      float   (bucket tolerance – percent, default 0.05)
#     order:          asc | desc | rand   (bucket ordering)
# """
# from __future__ import annotations
# import math, random, logging
# from typing import List, Tuple, Optional
# import pytorch_lightning as pl
# import torch
# from hydra.utils import instantiate
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader, BatchSampler

# log = logging.getLogger(__name__)

# # ----------------------------- collator -------------------------------- #
# def max_collator(
#     batch: List[Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]]
# ):
#     noisy, clean, txt, mask = zip(*batch)
#     # C,T → T,C for pad_sequence
#     noisy_tc = [x.transpose(0, 1) for x in noisy]
#     clean_tc = [x.transpose(0, 1) for x in clean]
#     noisy_pad = pad_sequence(noisy_tc,  batch_first=True)
#     clean_pad = pad_sequence(clean_tc,  batch_first=True)
#     mask_pad  = pad_sequence(mask,     batch_first=True)
#     noisy_b = noisy_pad.permute(0, 2, 1).contiguous()
#     clean_b = clean_pad.permute(0, 2, 1).contiguous()
#     log.debug(
#         f"[collate] B={len(batch)} max_len={noisy_b.shape[-1]} "
#         f"pad={100*(1-mask_pad.sum()/mask_pad.numel()):4.1f}%"
#     )
#     return noisy_b, clean_b, list(txt), mask_pad


# # ------------------------ samplers for bucket modes ------------------- #
# class _BucketSampler(BatchSampler):
#     """Shared parent for constant‑BS and budget samplers."""
#     def __init__(
#         self,
#         lengths: List[int],
#         fs: int,
#         *,
#         width_sec: Optional[float],
#         width_pct: float,
#         order: str,
#     ):
#         self.width_frames = int(width_sec * fs) if width_sec else None
#         ids = sorted(range(len(lengths)), key=lengths.__getitem__)
#         if order == "desc": ids.reverse()
#         elif order == "rand": random.shuffle(ids)
#         self.lengths = lengths
#         self.ids = ids

#     # ---- helper ------------------------------------------------------- #
#     def _same_bucket(self, a: int, b: int) -> bool:
#         la, lb = self.lengths[a], self.lengths[b]
#         if self.width_frames is not None:
#             return abs(la - lb) <= self.width_frames
#         return abs(la - lb) <= self.width_pct * la


# class BucketBatchSampler(_BucketSampler):
#     """constant *batch size*, clips within width_sec / width_pct."""
#     def __init__(self, lengths, batch_size, **kw):
#         super().__init__(lengths, **kw)
#         self.bs = batch_size
#         # make buckets
#         buckets, cur = [], [self.ids[0]]
#         for i in self.ids[1:]:
#             if not self._same_bucket(i, cur[0]):
#                 buckets.append(cur); cur = [i]
#             else:
#                 cur.append(i)
#         buckets.append(cur)
#         self.buckets = buckets

#     def __iter__(self):
#         for b in self.buckets:
#             random.shuffle(b)
#             for i in range(0, len(b), self.bs):
#                 yield b[i : i + self.bs]

#     def __len__(self):
#         return sum(math.ceil(len(b) / self.bs) for b in self.buckets)


# class VariableBatchSampler(_BucketSampler):
#     """constant *total length* budget per batch."""
#     def __init__(self, lengths, budget_frames, **kw):
#         super().__init__(lengths, **kw)
#         self.budget = budget_frames
#         # build simple ascending buckets for speed
#         self.buckets, cur = [], [self.ids[0]]
#         for i in self.ids[1:]:
#             if not self._same_bucket(i, cur[0]):
#                 self.buckets.append(cur); cur = [i]
#             else:
#                 cur.append(i)
#         self.buckets.append(cur)

#     def __iter__(self):
#         for b in self.buckets:
#             random.shuffle(b)
#             batch, tot = [], 0
#             for i in b:
#                 L = self.lengths[i]
#                 if tot + L > self.budget and batch:
#                     yield batch; batch, tot = [], 0
#                 batch.append(i); tot += L
#             if batch: yield batch

#     def __len__(self):
#         return sum( math.ceil( sum(self.lengths[i] for i in b) / self.budget)
#                     for b in self.buckets )

# # ------------------------- Lightning DataModule ----------------------- #
# class DataModule(pl.LightningDataModule):
#     # --- constructor --------------------------------------------------- #
#     def __init__(self, train, val, test, datasets, **cfg):
#         super().__init__()
#         self.split_cfg = dict(train=train, val=val, test=test)
#         self.ds_cfgs   = datasets
#         # generic hyper‑params (mode, fixed_len_sec, batch_size, …)
#         self.h = cfg
#         self.mode = cfg.get("mode", "fixed")

#     # --- instantiate datasets ----------------------------------------- #
#     def setup(self, stage=None):
#         self.datasets = {}
#         for split in ["train", "val", "test"]:
#             ds_name = self.split_cfg[split].dataset
#             ds_conf = self.ds_cfgs[ds_name]

#             # add fixed_len only for train & mode=fixed
#             if split == "train" and self.mode == "fixed":
#                 ds_conf = {**ds_conf, "fixed_len_sec": self.h["fixed_len_sec"]}

#             self.datasets[split] = instantiate(ds_conf, _recursive_=False)

#     # --- private helper to choose proper loader ----------------------- #
#     def _train_loader(self):
#         ds = self.datasets["train"]
#         h  = self.h
#         if self.mode == "fixed":
#             return DataLoader(
#                 ds,
#                 collate_fn=max_collator,
#                 **self.split_cfg["train"].dl_opts,
#             )

#         if self.mode == "bucket_bs":
#             sampler = BucketBatchSampler(
#                 ds.lengths,
#                 batch_size=h["batch_size"],
#                 fs=ds.fs,
#                 width_sec=h.get("width_sec"),
#                 width_pct=h.get("width_pct", 0.05),
#                 order=h.get("order", "asc"),
#             )
#             return DataLoader(ds, batch_sampler=sampler, collate_fn=max_collator)

#         if self.mode == "bucket_budget":
#             sampler = VariableBatchSampler(
#                 ds.lengths,
#                 budget_frames=int(h["budget_sec"] * ds.fs),
#                 fs=ds.fs,
#                 width_sec=h.get("width_sec"),
#                 width_pct=h.get("width_pct", 0.05),
#                 order=h.get("order", "asc"),
#             )
#             return DataLoader(ds, batch_sampler=sampler, collate_fn=max_collator)

#         raise ValueError(f"Unknown mode {self.mode}")

#     # ---------------- public loaders ---------------------------------- #
#     def train_dataloader(self): return self._train_loader()
#     def val_dataloader  (self):
#         return DataLoader(
#             self.datasets["val"],
#             collate_fn=max_collator,
#             **self.split_cfg["val"].dl_opts,
#         )
#     def test_dataloader (self):
#         return DataLoader(
#             self.datasets["test"],
#             collate_fn=max_collator,
#             **self.split_cfg["test"].dl_opts,
#         )
