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

import json, datetime                            # NEW

import torch.distributed as dist   
from torch.utils.data import get_worker_info


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
        

        
        # isolate every rank in its own directory
        rank = dist.get_rank() if dist.is_initialized() else 0

        for s, d in self._dump_cfg.items():
            d["dir"] = d["dir"] / f"rank{rank}"        # rank-specific path
            print(f"Debug dump for {s} (rank {rank}): {d['dir']}")
            for sub in ("clean", "noisy", "txt"):
                (d["dir"]/sub).mkdir(parents=True, exist_ok=True)
        
        
        
    def _make_collator(self, split: str):
        dump = self._dump_cfg[split]

        def collate(batch):
            # noisy, clean, txt, mask = zip(*batch)
            
            # ---------- unpack, supporting optional meta -------------
            if len(batch[0]) == 5:                          # new return
                noisy, clean, txt, mask, meta = zip(*batch)
            else:                                           # old return
                noisy, clean, txt, mask = zip(*batch)
                meta = (None,) * len(noisy)                 # <- define!
            
            
            L = max(x.shape[-1] for x in noisy)

            pad = lambda t: torch.nn.functional.pad(t, (0, L - t.shape[-1]))
            noisy = torch.stack([pad(x) for x in noisy])
            clean = torch.stack([pad(x) for x in clean])
            mask  = torch.stack([pad(m) for m in mask])

            # -------- optional on-the-fly export -----------------
            # if dump["dir"] and dump["cnt"] < dump["max"]:
            #     sr = 16_000
            #     rank = dist.get_rank() if dist.is_initialized() else 0 
            #     for i in range(noisy.shape[0]):
            #         # uid = f"b{dump['cnt']:02d}_{i:02d}"
            #         uid = f"r{rank}_b{dump['cnt']:02d}_{i:02d}"    
            #         torchaudio.save(dump["dir"] / "noisy" / f"{uid}.wav", noisy[i].cpu(), sr, backend="soundfile")
            #         torchaudio.save(dump["dir"] / "clean" / f"{uid}.wav", clean[i].cpu(), sr, backend="soundfile")
            #         (dump["dir"] / "txt" / f"{uid}.txt").write_text(txt[i])
                    
            #         # ---------- NEW: append one JSON line -------------
            #         log_line = {
            #             "batch": dump["cnt"],
            #             "rank":  rank,
            #             "index": i,
            #             "ts_ms": int(datetime.datetime.utcnow()
            #                          .timestamp() * 1000),
            #         }
            #         if meta[i] is not None:
            #             log_line.update(meta[i])
            #         (dump["dir"] / "log.txt").open("a").write(
            #             json.dumps(log_line) + "\n"
            #         )

            #     dump["cnt"] += 1
            
            # # -------- optional on-the-fly export -----------------
            # if dump["dir"] and dump["cnt"] < dump["max"]:          # limit = batches
            #     sr   = 16_000
            #     rank = dist.get_rank() if dist.is_initialized() else 0
                
            log_lines = []                               # ← buffer
                # -------- optional on-the-fly export -----------------
            # if dump["dir"] and dump["cnt"] < dump["max"]:
            winfo = get_worker_info()
            wid   = winfo.id if winfo else 0
            # if wid == 0 and dump["dir"] and dump["cnt"] < dump["max"]:   # ← NEW gate
            if dump["dir"] and dump["cnt"] < dump["max"]:      # ← gate removed
        
                # --- claim a unique batch number atomically
                batch_id = dump["cnt"]
                dump["cnt"] += 1          # do this *before* any disk I/O

                sr   = 16_000
                rank = dist.get_rank() if dist.is_initialized() else 0
                
                
                for i in range(noisy.shape[0]):
                    # uid   = f"r{rank}_b{dump['cnt']:02d}_{i:02d}"
                    uid   = f"r{rank}_w{wid}_b{batch_id:02d}_{i:02d}"        # worker tag
                
                for i in range(noisy.shape[0]):
                    # uid   = f"r{rank}_b{batch_id:02d}_{i:02d}"    
                    uid   = f"r{rank}_w{wid}_b{batch_id:02d}_{i:02d}"
                
                    fn_n  = dump["dir"] / "noisy"  / f"{uid}.wav"
                    fn_c  = dump["dir"] / "clean"  / f"{uid}.wav"
                    fn_t  = dump["dir"] / "txt"    / f"{uid}.txt"
                    
                                    
                    # torchaudio.save(fn_n, noisy[i].cpu(), sr, backend="soundfile")
                    # torchaudio.save(fn_c, clean[i].cpu(), sr, backend="soundfile")
                    # fn_t.write_text(txt[i])

                    # # ---- one JSON line per *saved* sample -------------
                    # meta_line = {
                    #     "uid"      : uid,
                    #     "batch"    : dump["cnt"],
                    #     "rank"     : rank,
                    #     "noisy_wav": fn_n.name,
                    #     "clean_wav": fn_c.name,
                    #     "txt_file" : fn_t.name,
                    #     "ts_ms"    : int(datetime.datetime.utcnow().timestamp()*1000),
                    # }
                    # (dump["dir"] / "log.txt").open("a").write(json.dumps(meta_line)+"\n")
                    
                    
                    if not fn_n.exists():                      # save ONLY once
                        torchaudio.save(fn_n, noisy[i].cpu(), sr, backend="soundfile")
                        torchaudio.save(fn_c, clean[i].cpu(), sr, backend="soundfile")
                        fn_t.write_text(txt[i])

                        # meta_line = {
                        #     "uid"      : uid,
                        #     "noisy_wav": fn_n.name,
                        #     "clean_wav": fn_c.name,
                        #     "txt_file" : fn_t.name,
                        #     "batch"    : dump["cnt"],
                        #     "rank"     : rank,
                        #     "ts_ms"    : int(datetime.datetime.utcnow().timestamp()*1000),
                        # }
                        
                        meta_line = {
                            "uid"      : uid,
                            "batch"    : batch_id,
                            "rank"     : rank,
                            "noisy_wav": fn_n.name,
                            "clean_wav": fn_c.name,
                            "txt_file" : fn_t.name,
                            "ts_ms"    : int(datetime.datetime.utcnow().timestamp()*1000),
                            }
                        
                        
                        
                        
                        if meta[i] is not None:                # add blocks, fn src …
                            meta_line.update(meta[i])
                            
                            # ---- NEW: source‑clean duration (seconds) ------------
                            if "clean_fn" in meta[i]:          # absolute path stored by dataset
                                info = torchaudio.info(meta[i]["clean_fn"])
                                meta_line["clean_dur_s"] = info.num_frames / info.sample_rate
                                
                            
                        log_lines.append(meta_line)

                if log_lines:                                   # write once/batch
                    log_path = dump["dir"] / "log.txt"
                    with log_path.open("a") as fh:
                        for line in log_lines:
                            fh.write(json.dumps(line) + "\n")
                    

                dump["cnt"] += 1                                   # advance *once per batch*
            
            
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
