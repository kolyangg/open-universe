# universe_gan_NS2_adj.py
# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
Merged UNIVERSE(++) with HiFi-GAN loss and optional text support.

- Base structure comes from old universe_gan_NS.py (discriminator/generator logic).
- Adds text features & debug prints from the new version (universe_gan_NS2.py).
- If text is absent/empty, we follow the exact old training process.
- If text is present, we enable new text logic & debugging, without disturbing the old baseline flow.
"""

import itertools
import logging

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .. import bigvgan as gan
from .blocks import PReLU_Conv
# Import the "Universe" or any base class you use:
# from .m_universe_NS_adj3_fix import Universe 
from .m_universe_NS_adj3_fix_map_orig import Universe 

log = logging.getLogger(__name__)


def instantiate_scheduler(config_scheduler, optimizer):
    if config_scheduler is None:
        return None
    scheduler = OmegaConf.to_container(config_scheduler, resolve=True)
    lr_sch_kwargs = scheduler.pop("scheduler")
    scheduler["scheduler"] = instantiate(
        {**lr_sch_kwargs, **{"optimizer": optimizer}}, _recursive_=False
    )
    return scheduler


def step_scheduler(
    scheduler, interval, frequency, epoch, step, is_last_batch, loss_val=None
):
    if scheduler is None:
        return
    on_epoch = interval == "epoch" and is_last_batch and (epoch + 1) % frequency == 0
    on_step = interval == "step" and (step + 1) % frequency == 0
    if on_epoch or on_step:
        if loss_val is not None:
            scheduler.step(loss_val)
        else:
            scheduler.step()


class UniverseGAN(Universe):
    def __init__(
        self,
        fs,
        normalization_norm,
        score_model,
        condition_model,
        diffusion,
        losses,
        training,
        validation,
        optimizer,
        scheduler,
        grad_clipper,
        transform=None,
        normalization_kwargs={},
        detach_cond=False,
        edm=None,
        use_text_in_val = True
    ):
        """
        This merges the "old" UniverseGAN code with the new text logic.
        If text is absent, training is identical to the old code.
        If text is present, we do new text features and debug prints.
        """
        super().__init__(
            fs,
            normalization_norm,
            score_model,
            condition_model,
            diffusion,
            losses,
            training,
            validation,
            optimizer,
            scheduler,
            grad_clipper,
            transform=transform,
            normalization_kwargs=normalization_kwargs,
            detach_cond=detach_cond,
            edm=edm
        )

        self.use_text_in_val = use_text_in_val # NEW FOR TEXT

        # For GAN training, we disable automatic optimization.
        self.automatic_optimization = False

        # For logging
        self.log_kwargs = dict(
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )

        # [DEBUG] If you want to control text usage externally, you could do:
        # self.have_text = True  # or read from config
        # but we'll detect it per-batch in training_step below.
        
        self.debug_print= False

    def init_losses(self, score_model, condition_model, losses, training):
        """Initialize the GAN losses here."""
        # Define discriminators. MPD is used by default.
        self.loss_mpd = gan.MultiPeriodDiscriminator(losses.multi_period_discriminator)

        # Define additional discriminators. BigVGAN uses MRD as default.
        self.loss_mrd = gan.MultiResolutionDiscriminator(
            losses.multi_resolution_discriminator
        )

        if losses.use_signal_decoupling:
            self.signal_decoupling_layer = PReLU_Conv(
                self.n_channels,
                1,
                kernel_size=3,
                padding="same",
                act_type=losses.get("signal_decoupling_act", None),
            )
        else:
            self.signal_decoupling_layer = None

        self.loss_score = instantiate(losses.score_loss, _recursive_=False)
        self.disc_freeze_step = losses.get("disc_freeze_step", 0)

        if losses.get("aux_loss", None) is not None:
            self.loss_aux = instantiate(losses.aux_loss, _recursive_=False)
        else:
            self.loss_aux = None

    def model_parameters(self):
        # Same as old code, excluding discriminator from main param set
        params = itertools.chain(
            self.get_score_model().parameters(), self.condition_model.parameters()
        )
        if self.signal_decoupling_layer is not None:
            params = itertools.chain(params, self.signal_decoupling_layer.parameters())
        return params

    def aux_to_wav(self, y_aux):
        # same as old code
        if self.signal_decoupling_layer is not None:
            return self.signal_decoupling_layer(y_aux)
        else:
            return y_aux

    def training_step(self, batch, batch_idx):
        """
        If text is present (batch has >=3 items and the 3rd is non-empty), 
        we do the "new" text logic. Otherwise we do old code's baseline flow.
        """
        opt_score, opt_disc = self.optimizers()

            
        
        try:                                    # lightning may now return 2 or 3 schedulers
            schedulers   = self.lr_schedulers()     # list-like
            sch_score    = schedulers[0]            # keep original names
            sch_disc     = schedulers[1]
            has_schedulers = True
        except TypeError:                       # no schedulers configured
            schedulers     = []
            has_schedulers = False

        # Expected layouts now:
        #  • (mix, tgt, text, mask)      ← normal
        #  • (mix, tgt,   –,  mask)      ← no‑text training
        #  • legacy 3‑tuple is still OK.
        if len(batch) == 4:                          # ← NEW
            mix_raw, target_raw, text, mask = batch
            text_str_list = text if isinstance(text, list) else [text]
            # For backward compatibility, we keep the original approach:
            target_original = target_raw
        elif len(batch) == 3 and isinstance(batch[2], (str, list)):
            mix_raw, target_raw, text = batch
            text_str_list = text if isinstance(text, list) else [text]
            mask = None
            # For backward compatibility, we keep the original approach:
            target_original = target_raw
        elif len(batch) == 3:                        # mix, tgt, mask  (no text)
            mix_raw, target_raw, mask = batch
            text = []
            # For backward compatibility, we keep the original approach:
            target_original = target_raw
        else:                                        # legacy 2‑tuple
            mix_raw, target_raw = batch
            text, mask = [], None
            # For backward compatibility, we keep the original approach:
            target_original = target_raw

        
        
        if getattr(self.train_kwargs, "dynamic_mixing", False):
            noise = mix_raw - target_raw
            perm = torch.randperm(noise.shape[0])
            mix_raw = target_raw + noise[perm, ...]


        
        ##### FIX 03 MAY #####
        
        # --- keep *target_original* UN‑normalised (old behaviour) -------------
        # target_original = target_raw.clone()
        target_original = target_raw #### FIX 04 MAY

        # normalise only mix & main target
        (mix, target), *stats = self.normalize_batch(
            (mix_raw, target_raw), norm=self.normalization_norm
        )
        
        ##### FIX 03 MAY #####
        
        # ---------- ❶  mask out padded frames early ------------------------
        if mask is not None:
            mask_c = mask.unsqueeze(1)          # [B,1,T]
            mix            = mix   * mask_c
            target         = target * mask_c
            target_original= target_original * mask_c
        

        if self.transform is not None:
            mix = self.transform(mix)
            target = self.transform(target)

        # sample sigma
        sigma, _ = self.sample_sigma(mix, self.train_kwargs.time_sampling, 0.0, 1.0)
        
        # sample the noise and create the target
        z = target.new_zeros(target.shape).normal_()
        # print(f'[DEBUG MASK]: {mask}')
        if mask is not None:                    # do NOT inject noise in padding
           z = z * mask_c
        
        
        x_t = target + sigma[:, None, None] * z

        # ---------------------------
        # Now the main difference: if text is present => pass text to condition_model
        # so it can do FiLM/cross-attn and debug prints.
        # If text is empty => old code path
        # ---------------------------
        use_text = any(t.strip() for t in text_str_list)
        if use_text:
            # "new" text approach with debug prints
            result = self.condition_model(mix, text=text_str_list, train=True, mask = mask) ### 01 MAY: added mask
            if len(result) == 4:
                cond, y_est, _, text_metrics = result
                # Log text debug metrics just like in new code
                for k, v in text_metrics.items():
                    if isinstance(v, (int, float)):
                        self.log(f"text_checks/{k}", v, batch_size=mix.shape[0], **self.log_kwargs)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"text_checks/top_attended_{i}", pos, batch_size=mix.shape[0], **self.log_kwargs)
            
            elif len(result) == 5:
                cond, y_est, _, text_metrics1, text_metrics2 = result
                # Log text debug metrics just like in new code
                for k, v in text_metrics1.items():
                    if isinstance(v, (int, float)):
                        self.log(f"text_checks1/{k}", v, batch_size=mix.shape[0], **self.log_kwargs)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"text_checks1/top_attended_{i}", pos, batch_size=mix.shape[0], **self.log_kwargs)
                
                for k, v in text_metrics2.items():
                    if isinstance(v, (int, float)):
                        self.log(f"text_checks2/{k}", v, batch_size=mix.shape[0], **self.log_kwargs)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"text_checks2/top_attended_{i}", pos, batch_size=mix.shape[0], **self.log_kwargs)
            else:
                cond, y_est, _ = result
            # print("[DEBUG] Text-based conditioning active in training_step.")
        else:
            # old code: just call condition_model with no text
            # cond, y_est, _ = self.condition_model(mix, train=True, mask = mask) ### 01 MAY: added mask
            cond, y_est, _ = self.condition_model(mix, train=True) ### 04 MAY: removed mask if no text (TBC)
            if self.debug_print:
                print("[DEBUG] No text => old conditioning path in training_step.")

        if self.detach_cond:
            cond = [c.detach() for c in cond]

        # compute score
        score = self.score_model(x_t, sigma, cond)

        # decouple the signal estimate
        if self.signal_decoupling_layer is not None:
            y_est = self.signal_decoupling_layer(y_est)

        # invert transform
        if self.transform is not None:
            y_est = self.transform(y_est, inv=True)

        # use a simple mel loss to regularize training
        mel_y_est = self.condition_model.input_mel.compute_mel_spec(y_est)
        mel_target = self.condition_model.input_mel.compute_mel_spec(target_original)
        
        ### 19 APR ADD
        # ---------- mask generator output & targets ------------------------
        if mask is not None:
            y_est            = y_est            * mask_c
            target_original  = target_original  * mask_c

        # (keep 'cond' untouched – its convolutions already saw masked mix)


        # -------------------------------
        # Discriminator optimization
        # -------------------------------
        opt_disc.zero_grad()

        # 1) MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.loss_mpd(target_original, y_est.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = gan.discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )
        # 2) MRD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.loss_mrd(target_original, y_est.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = gan.discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        loss_disc = loss_disc_s + loss_disc_f

        if self.global_step >= self.disc_freeze_step:
            self.manual_backward(loss_disc)
            grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
                self.loss_mpd.parameters(), self.grad_clip_vals.mpd
            )
            grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
                self.loss_mrd.parameters(), self.grad_clip_vals.mrd
            )
            opt_disc.step()
            # step schedulers if needed
            # if has_schedulers:
            #     self.step_schedulers(sch_score, sch_disc)
            if has_schedulers:
                self.step_schedulers(*schedulers)
        else:
            grad_norm_mpd = 0.0
            grad_norm_mrd = 0.0

        if torch.isnan(loss_disc):
            print(f"Found NaN!! Please investigate. {loss_disc_s=} {loss_disc_f=}")
            breakpoint()

        # -------------------------------
        # Generator / score optimization
        # -------------------------------
        opt_score.zero_grad()

        # 1. Score loss
        l_score = self.loss_score(sigma[..., None, None] * score, -z)
        
        if mask is not None:
            m = mask_c                         # broadcast helper
            l_score = self.loss_score(sigma[...,None,None]*score*m, -z*m)
        else:
            l_score = self.loss_score(sigma[...,None,None]*score, -z)
        
        # 2. L1 Mel-Spectrogram Loss
        loss_mel = torch.nn.functional.l1_loss(mel_y_est, mel_target)
        
        # 3. Auxiliary loss
        if self.loss_aux is not None:
            # optional linear warmup of the loss if it is introduced late
            aux_loss_schedule = self.losses_kwargs.get("aux_loss_schedule", {})
            start_step = aux_loss_schedule.get("start_step", 0)
            warmup_steps = aux_loss_schedule.get("warmup_steps", 0)
            if self.global_step >= start_step:
                w = (
                    min(1.0, (self.global_step - start_step) / warmup_steps)
                    if warmup_steps > 0
                    else 1.0
                )
                loss_aux = w * self.loss_aux(y_est, target_original)
            else:
                loss_aux = 0.0
        else:
            loss_aux = 0.0

        # 4) MPD gen
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.loss_mpd(
            target_original, y_est
        )
        loss_fm_f = gan.feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, losses_gen_f = gan.generator_loss(y_df_hat_g)

        # 5) MRD gen
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.loss_mrd(
            target_original, y_est
        )
        loss_fm_s = gan.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = gan.generator_loss(y_ds_hat_g)

        loss_gen = (
            l_score * self.losses_kwargs.weights.score
            + loss_mel * self.losses_kwargs.weights.mel_l1
            + loss_aux * self.losses_kwargs.weights.get("aux", 1.0)
        )
        if self.global_step >= self.disc_freeze_step:
            loss_gen = loss_gen + loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f

        self.manual_backward(loss_gen)

        grad_norm_score = torch.nn.utils.clip_grad_norm_(
            self.get_score_model().parameters(), self.grad_clip_vals.score
        )
        grad_norm_cond = torch.nn.utils.clip_grad_norm_(
            self.condition_model.parameters(), self.grad_clip_vals.cond
        )
        opt_score.step()
        
        #############################################################
        # TEXT-LR LOGGING  (new)
        #############################################################
        text_lr = None
        for pg in opt_score.param_groups:
            if any(id(p) in self.text_ids for p in pg["params"]):
                text_lr = pg["lr"]; break
        if text_lr is not None:
            self.log("text_checks/lr", text_lr,
                    batch_size=mix.shape[0], **self.log_kwargs)
# --------------------------------------------------------
        #############################################################

        if self.ema is not None:
            self.ema.update(self.model_parameters())

        # if has_schedulers:
        #     self.step_schedulers(sch_score, sch_disc)
        if has_schedulers:
                self.step_schedulers(*schedulers)

        # every few steps, we log stuff
        self.log(
            "train/loss_disc",
            loss_disc,
            batch_size=mix.shape[0],
            prog_bar=True,
            **self.log_kwargs,
        )
        self.log(
            "train/loss_mpd", loss_disc_f, batch_size=mix.shape[0], **self.log_kwargs
        )
        self.log(
            "train/loss_mrd", loss_disc_s, batch_size=mix.shape[0], **self.log_kwargs
        )

        self.log(
            "train/loss_gen",
            loss_gen,
            prog_bar=True,
            batch_size=mix.shape[0],
            **self.log_kwargs,
        )
        self.log("train/score", l_score, batch_size=mix.shape[0], **self.log_kwargs)
        self.log(
            "train/signal_mel", loss_mel, batch_size=mix.shape[0], **self.log_kwargs
        )
        if self.loss_aux is not None:
            self.log(
                "train/signal_aux", loss_aux, batch_size=mix.shape[0], **self.log_kwargs
            )
        self.log("train/mrd_fm", loss_fm_s, batch_size=mix.shape[0], **self.log_kwargs)
        self.log("train/mpd_fm", loss_fm_f, batch_size=mix.shape[0], **self.log_kwargs)
        self.log(
            "train/mrd_gen", loss_gen_s, batch_size=mix.shape[0], **self.log_kwargs
        )
        self.log(
            "train/mpd_gen", loss_gen_f, batch_size=mix.shape[0], **self.log_kwargs
        )
        self.log("grad/score", grad_norm_score, **self.log_kwargs)
        self.log("grad/cond", grad_norm_cond, **self.log_kwargs)
        self.log("grad/mpd", grad_norm_mpd, **self.log_kwargs)
        self.log("grad/mrd", grad_norm_mrd, **self.log_kwargs)

        if torch.isnan(loss_gen):
            print(
                f"Found NaN!! Please investigate. {l_score=} {loss_mel=} "
                f"{loss_gen_s=} {loss_gen_f=} {loss_fm_s=} {loss_fm_f=}"
            )
            breakpoint()

        # Return a dict or just the final loss for logging
        return {"loss": loss_gen}

    def step_schedulers(self, sch_score, sch_disc, *extra_scheds):
        # step the schedulers
        step_scheduler(
            sch_score,
            self.schedule_kwargs.generator.interval,
            self.schedule_kwargs.generator.frequency,
            self.current_epoch,
            self.global_step,
            self.trainer.is_last_batch,
            loss_val=None,
        )
        step_scheduler(
            sch_disc,
            self.schedule_kwargs.discriminator.interval,
            self.schedule_kwargs.discriminator.frequency,
            self.current_epoch,
            self.global_step,
            self.trainer.is_last_batch,
            loss_val=None,
        )
        
        # --- NEW: any additional schedulers (text, etc.) -------------
        for sch in extra_scheds:
            # use a simple “every step” policy unless you later expose
            # dedicated interval / frequency in the config
            step_scheduler(
                sch,
                interval="step",
                frequency=1,
                epoch=self.current_epoch,
                step=self.global_step,
                is_last_batch=self.trainer.is_last_batch,
                loss_val=None,
            )

    def configure_optimizers(self):
        # generator
        opt_kwargs = OmegaConf.to_container(self.opt_kwargs.generator, resolve=True)

        # We can have a list of keywords to exclude from weight decay
        weight_decay = opt_kwargs.pop("weight_decay", 0.0)
        wd_exclude_list = opt_kwargs.pop("weight_decay_exclude", [])
        
        ### NEW 30 APR ###
        text_lr_scale    = getattr(self.condition_model, "text_lr_scale", 1.0)
        base_lr          = opt_kwargs["lr"]          # keep for later
        ### NEW 30 APR ###
        

        def pick_excluded(name):
            return any([kw in name for kw in wd_exclude_list])


        #### NEW 30 APR ####
        excluded, others = [], []
        for submod in [self.get_score_model(), self.condition_model]:
            for name, p in submod.named_parameters():
                if not p.requires_grad:
                    continue
                (excluded if pick_excluded(name) else others).append(p)

        use_text_branch = getattr(self.condition_model, "text_conditioner", None) is not None

        text_params = []
        if use_text_branch:
            text_params = [p for n, p in self.condition_model.named_parameters()
                           if n.startswith("text_conditioner") and p.requires_grad]
        self.text_ids = {id(p) for p in text_params}

        others   = [p for p in others   if id(p) not in self.text_ids]
        excluded = [p for p in excluded if id(p) not in self.text_ids]

        param_groups = [
            {"params": excluded},
            {"params": others, "weight_decay": weight_decay},
        ]
        if text_params:          # only add if it is really used
            param_groups.append(
                {"params": text_params, "lr": base_lr * text_lr_scale}
            )
        ### 03 MAY FIX (ADDED 30 APR) ###
        

        log.info(f"Generator LR={base_lr:.4g} | text-lr scale={text_lr_scale}")

        optimizer_gen = instantiate({**opt_kwargs, "params": param_groups},
                                    _recursive_=False, _convert_="all")
        
        
        
        #### NEW 30 APR ####
        

        # optimizer discriminator
        params_disc = list(self.loss_mrd.parameters()) + list(
            self.loss_mrd.parameters()
        )
        optimizer_disc = instantiate(
            self.opt_kwargs.discriminator, params=params_disc, _recursive_=False
        )

        self.grad_clip_vals = self.opt_kwargs.grad_clip_vals
        self.grad_clipper = None
        
        if self.schedule_kwargs is not None:
            scheduler_gen = instantiate_scheduler(
                self.schedule_kwargs.get("generator"), optimizer_gen
            )
            scheduler_disc = instantiate_scheduler(
                self.schedule_kwargs.get("discriminator"), optimizer_disc
            )
            
            
            sched_list = [scheduler_gen, scheduler_disc]
            sched_cfg_text = (
                self.schedule_kwargs.get("text") if text_params else None
            )
            if sched_cfg_text is not None:
                from torch.optim.lr_scheduler import LambdaLR

                total_iters  = sched_cfg_text["scheduler"]["total_iters"]
                start_factor = sched_cfg_text["scheduler"]["start_factor"]
                end_factor   = sched_cfg_text["scheduler"].get("end_factor", 1.0)

                def linear_warmup(step):
                    if step >= total_iters:
                        return 1.0
                    alpha = step / float(total_iters)
                    return start_factor + alpha * (end_factor - start_factor)

                # λ‑list length == #param‑groups
                lambdas = [
                    lambda step: 1.0,      # group‑0  (excluded params)
                    lambda step: 1.0,      # group‑1  (weight‑decay params)
                    linear_warmup          # group‑2  (text params)
                ]

                scheduler_text = LambdaLR(optimizer_gen, lr_lambda=lambdas)
                
                

                sched_list.append(
                    dict(
                        scheduler=scheduler_text,
                        interval=sched_cfg_text.get("interval", "step"),
                        frequency=sched_cfg_text.get("frequency", 1),
                        # No `param_group_indices` needed – groups 0 & 1 keep LR
                    )
                )
            

            return [optimizer_gen, optimizer_disc], sched_list
            
           # return [optimizer_gen, optimizer_disc], [scheduler_gen, scheduler_disc]
        else:
            return [optimizer_gen, optimizer_disc]

    def on_after_backward(self):
        pass