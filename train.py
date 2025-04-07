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

    wandb_logger = pl_loggers.WandbLogger(
        # project="universe",
        project="universe_small",
        name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
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
