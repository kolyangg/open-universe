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
# Main training script.

This script will read a hydra config and start the training of a model.
For each run, an experiment directory named `exp/<experiment_name>/<datetime>`
containing config, checkpoints, and tensorboard logs will be created.

Author: Robin Scheibler (@fakufaku)
"""

print("starting import")
import logging
import os

import hydra
print('imported hydra')
import pytorch_lightning as pl
print('imported py lightning')
import torch
print('imported torch')
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
from omegaconf.omegaconf import open_dict
from pytorch_lightning import loggers as pl_loggers
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

print('imported until open_univ')
from open_universe import utils
print('imported utils')

import pathlib
import re


from rsync.cloud_sync import RsyncBackup   ### 02 May - added rsync

log = logging.getLogger(__name__)


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    hydra_conf = HydraConfig().get()
    exp_name = hydra_conf.run.dir

    if utils.ddp.is_rank_zero():
        log.info(f"Start experiment: {exp_name}")
    else:
        os.chdir(hydra.utils.get_original_cwd())

    rank, world_size, worker, num_workers = utils.ddp.pytorch_worker_info()
    log.info(f"{rank=}/{world_size} {worker=}/{num_workers} PID={os.getpid()}")

    pl.seed_everything(cfg.seed)

    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("medium")

    checkpoint_dir = os.path.abspath("checkpoints/universe/exper")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[DEBUG] 🔍 Checkpoints will be saved in: {checkpoint_dir}")

    val_loss_name = f"{cfg.model.validation.main_loss}"
    loss_name = val_loss_name.split("/")[-1]

    # wandb_logger = pl_loggers.WandbLogger(
    #     # project="universe",
    #     # project="universe_small",
    #     project="universe_4s",
    #     name=exp_name,
    #     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # )
    
    
    # ───────────────────────────────────────────────────────────
    #  Re‑use previous wandb run if ckpt_path given
    #  Algorithm (per user spec):
    #  1) start from ckpt_path
    #  2) go three parents up   →   <run_folder>/
    #  3) enter  <run_folder>/wandb/
    #  4) take the single sub‑folder name  run‑YYYYMMDD_HHMMSS‑<id>
    #  5) extract <id>  (text after last "-")
    # ───────────────────────────────────────────────────────────
    
    wandb_id = None
    if getattr(cfg, "ckpt_path", None):
        # make ckpt absolute w.r.t. original cwd (avoid ./exp/exp duplication)
        orig_cwd = hydra.utils.get_original_cwd()
        ckpt     = pathlib.Path(cfg.ckpt_path)
        if not ckpt.is_absolute():
            ckpt = pathlib.Path(orig_cwd) / ckpt
        ckpt = ckpt.expanduser().resolve()
        print(f"[DEBUG] resolved ckpt_path → {ckpt}")

        try:
            run_dir   = ckpt.parents[3]                               # step 2
            wandb_dir = run_dir / "wandb"                             # step 3
            subdirs   = [d for d in wandb_dir.iterdir() if d.is_dir()]
            # if serveal subidirs, take the one whose name starts with "run-"
            subdirs   = [d for d in subdirs if d.name.startswith("run-")]
            
            if len(subdirs) != 1:
                raise RuntimeError(f"expected 1 subdir in {wandb_dir}, found {len(subdirs)}")

            run_folder = subdirs[0].name                              # step 4
            wandb_id   = run_folder.rsplit("-", 1)[-1]                # step 5

            # sanity‑check pattern
            if not re.fullmatch(r"[A-Za-z0-9]+", wandb_id):
                raise RuntimeError(f"run folder '{run_folder}' does not end with id")
            print(f"[DEBUG] Continuing previous wandb run → id={wandb_id}")

        except Exception as e:
            print(f"[DEBUG] Cannot resume wandb run → {e}")
            wandb_id = None

    wandb_logger = pl_loggers.WandbLogger(
        # project="universe_4s",
        # project="univ_4gpu",
        project="univ_2gpu_new",
        name=exp_name,
        id=wandb_id,                       # None ⇒ start new run
        resume="allow" if wandb_id else None,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
        
        
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=val_loss_name,
        mode=cfg.model.validation.main_loss_mode,
        filename="best-model",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        # RsyncBackup(run_dir=exp_name,             # ← NEW (no‑op if env not set)        ### NEW 02 MAY
        # remote_root=os.getenv("CLOUD_EXP_ROOT")),  
        RsyncBackup(remote_root=os.getenv("CLOUD_EXP_ROOT") or print("[RsyncBackup] CLOUD_EXP_ROOT not set – uploads DISABLED")) ### NEW 02 MAY
        
    ]

    print("Using the DCASE2020 SELD original dataset")
    log.info("create datalogger")

    dm = instantiate(cfg.datamodule, _recursive_=False)
    log.info(f"Create datamodule with training set: {cfg.datamodule.train.dataset}")

    log.info(f"Create new model {cfg.model._target_}")
    model = instantiate(cfg.model, _recursive_=False)

    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path is not None:
        ckpt_path = to_absolute_path(ckpt_path)
        with open_dict(cfg):
            cfg.trainer.num_sanity_val_steps = 0

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=wandb_logger)

    if cfg.train:
        log.info("start training")
        trainer.fit(model, dm, ckpt_path=ckpt_path)

        if trainer.is_global_zero:
            best_model_path = checkpoint_callback.best_model_path
            wandb_logger.experiment.save(best_model_path)
            print(f"[INFO] Uploaded best model checkpoint to Wandb: {best_model_path}")

    if cfg.test:
        try:
            log.info("start testing")
            trainer.test(model, dm, ckpt_path="best")
        except pl.utilities.exceptions.MisconfigurationException:
            log.info("test with current model value because no best model path is available")
            trainer.validate(model, dm)
            trainer.test(model, dm)


if __name__ == "__main__":
    print('starting')
    main()
