# universe_NS_adj.py
# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
The UNIVERSE(++) model with MDN loss

Merged version:
  - If no text in batch => old code exactly
  - If text present => new text path with debug prints
  - Preserves old code's MDN, normalization, validation flow, etc.
"""

import itertools
import logging
import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage

from ... import utils
from ...layers.dyn_range_comp import IdentityTransform
from .blocks import remove_weight_norm
from .mdn import MixtureDensityNetworkLoss

import wandb
import numpy as np

log = logging.getLogger(__name__)


def randn(x, sigma, rng=None):
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=rng)
    return noise * sigma[:, None, None]


# --- mask utilities -------------------------------------------------
def _apply_mask(wav, mask):
    if mask is None:
        return wav
    return wav * mask.unsqueeze(1)            # [B,1,T] → broadcast
def _downsample_mask(mask, factor):
    """average‑pool then binarise so it aligns with mel/latent length."""
    if mask is None:
        return None
    m = torch.nn.functional.avg_pool1d(mask.unsqueeze(1), factor, factor)
    return (m > 0.5).float()                  # shape [B,1,T//factor]



class Universe(pl.LightningModule):
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
        with_noise_target=False,
        detach_cond=False,
        edm=None,
    ):
        """
        If text isn't used, this code runs exactly like the old Universe class.
        If text is present, we do the new text path with debug prints.
        """
        super().__init__()
        self.save_hyperparameters()

        self.fs = fs
        self.normalization_norm = normalization_norm
        self.normalization_kwargs = normalization_kwargs
        self.with_noise_target = with_noise_target
        self.detach_cond = detach_cond

        self.opt_kwargs = optimizer
        self.schedule_kwargs = scheduler
        self.grad_clip_kwargs = grad_clipper

        self.diff_kwargs = diffusion
        self.losses_kwargs = losses
        self.val_kwargs = validation
        self.train_kwargs = training

        # *** ADDED: We'll dynamically detect text if a batch has >=3 items
        # but we also keep a "have_text" switch if needed.
        self.have_text = False  # This will be toggled when we see text in batch

        # optional EDM (unchanged from old code)
        if edm is not None:
            log.info("Use EDM network parameterization")
            self.edm_kwargs = edm
            # wrap the original score function
            self._edm_model = instantiate(score_model, _recursive_=False)
            self.score_model = self._edm_score_wrapper
            self.with_edm = True
        else:
            self.score_model = instantiate(score_model, _recursive_=False)
            self.with_edm = False

        # Condition model & standard parameters
        self.condition_model = instantiate(condition_model, _recursive_=False)
        self.n_channels = score_model.get("n_channels", 32)
        rate_factors = score_model.get("rate_factors", [2, 4, 4, 5])
        self.n_stages = len(rate_factors)
        self.latent_n_channels = 2**self.n_stages * self.n_channels
        self.tot_ds = math.prod(rate_factors)

        # Initialize losses
        self.init_losses(score_model, condition_model, losses, training)

        # MDN validation losses
        self.enh_losses = torch.nn.ModuleDict()
        for name, loss_args in self.val_kwargs.enh_losses.items():
            self.enh_losses[name] = instantiate(loss_args)

        self.denormalize_batch = utils.denormalize_batch

        # Transform
        if transform is None:
            self.transform = IdentityTransform()
        else:
            self.transform = instantiate(transform, _recursive_=False)

        # for moving average of weights
        # we exclude the loss parameters
        self.ema_decay = getattr(self.train_kwargs, "ema_decay", 0.0)
        log.info(f"Use EMA with decay {self.ema_decay}")
        if self.ema_decay > 0.0:
            self.ema = ExponentialMovingAverage(
                self.model_parameters(), decay=self.ema_decay
            )
            self._error_loading_ema = False
        else:
            self.ema = None
            self._error_loading_ema = False

    def model_parameters(self):
        return itertools.chain(self.get_score_model().parameters(), self.condition_model.parameters())

    def remove_weight_norm(self):
        remove_weight_norm(self)

    def init_losses(self, score_model, condition_model, losses, training):
        alpha_per_sample = losses.get("mdn_alpha_per_sample", False)
        log.info(f"Losses: Mixture density networks with {alpha_per_sample=}")
        """separate this init to allow to redefine in derived class"""

        cond_input_channels = getattr(condition_model, "input_channels", 1)
        num_targets = 2 if self.with_noise_target else 1

        if losses.weights.signal > 0.0:
            self.loss_signal = MixtureDensityNetworkLoss(
                est_channels=self.n_channels,
                tgt_channels=cond_input_channels * num_targets,
                n_comp=losses.mdn_n_comp,
                sampling_rate=self.fs // cond_input_channels,
                sample_len_s=training.audio_len,
                alpha_per_sample=alpha_per_sample,
            )
        else:
            self.loss_signal = None

        if losses.weights.latent > 0.0:
            self.loss_latent = MixtureDensityNetworkLoss(
                est_channels=self.latent_n_channels,
                tgt_channels=condition_model.n_mels * num_targets,
                n_comp=losses.mdn_n_comp,
                sampling_rate=self.fs // (cond_input_channels * self.tot_ds),
                sample_len_s=training.audio_len,
                alpha_per_sample=alpha_per_sample,
            )
        else:
            self.loss_latent = None

        self.loss_score = instantiate(losses.score_loss, _recursive_=False)

    def normalize_batch(self, batch, norm=None):
        if norm is None:
            norm = self.normalization_norm
        return utils.normalize_batch(batch, norm=norm, **self.normalization_kwargs)

    def _get_edm_weights(self, sigma):
        level_db = self.edm_kwargs.get(
            "data_level_db", self.normalization_kwargs.get("level_db", 0.0)
        )
        sigma_data = 10.0 ** (level_db / 20.0)
        sigma_norm = (sigma**2 + sigma_data**2) ** 0.5

        weights = {
            "skip": sigma_data**2 / (sigma**2 + sigma_data**2),
            "in": 1.0 / sigma_norm,
            "out": sigma * sigma_data / sigma_norm,
            "noise": self.edm_kwargs.noise,
        }

        return weights

    def get_score_model(self):
        if self.with_edm:
            return self._edm_model
        else:
            return self.score_model

    def _edm_score_wrapper(self, x, sigma, cond, with_speech_est=False):
        w = self._get_edm_weights(sigma)
        w_in = utils.pad_dim_right(w["in"], x)
        w_out = utils.pad_dim_right(w["out"], x)
        w_skip = utils.pad_dim_right(w["skip"], x)
        net_out = self._edm_model(w_in * x, w["noise"] * sigma, cond)
        speech_est = w_skip * x + w_out * net_out
        score = (speech_est - x) / utils.pad_dim_right(sigma, x) ** 2

        if with_speech_est:
            return score, speech_est
        else:
            return score

    def print_count(self):
        num_params_score_and_cond = utils.count_parameters(
            self.condition_model
        ) + utils.count_parameters(self.score_model)
        print(f"UNIVERSE number of parameters: {num_params_score_and_cond}")
        self.condition_model.print_count(indent=2)
        self.score_model.print_count(indent=2)

    def pad(self, x, pad=None):
        if pad is None:
            pad = self.tot_ds - x.shape[-1] % self.tot_ds
        x = torch.nn.functional.pad(x, (pad // 2, pad - pad // 2))
        return x, pad

    def unpad(self, x, pad):
        return x[..., pad // 2 : -(pad - pad // 2)]

    def aux_to_wav(self, y_aux):
        return y_aux

    # -------------------------------------------------------------------
    #   ENHANCE method: minimal addition to pass optional text
    # -------------------------------------------------------------------
    def enhance(
        self,
        mix,
        n_steps: Optional[int] = None,
        epsilon: Optional[float] = None,
        target: Optional[torch.Tensor] = None,
        fake_score_snr: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
        use_aux_signal: Optional[bool] = False,
        keep_rms: Optional[bool] = False,
        ensemble: Optional[int] = None,
        ensemble_stat: Optional[str] = "median",
        warm_start: Optional[int] = None,
        text: Optional[torch.Tensor] = None,  # <--- ADDED for text
        mask: Optional[torch.Tensor] = None,  # ← NEW
    ) -> torch.Tensor:
        """
        If text is provided, we call condition_model with text. If not, old path.
        """
        if epsilon is None:
            epsilon = self.diff_kwargs.epsilon
        if n_steps is None:
            n_steps = self.diff_kwargs.n_steps

        x_ndim = mix.ndim
        
        if mask is not None:
            mix = mix * mask.unsqueeze(1)          # zero padded region
        x_ndim = mix.ndim
        
        
        if x_ndim == 1:
            mix = mix[None, None, :]
        elif x_ndim == 2:
            mix = mix[:, None, :]
        elif x_ndim > 3:
            raise ValueError("The input should have at most 3 dimensions")

        mix_rms = mix.square().mean(dim=(-2, -1), keepdim=True).sqrt()

        if ensemble is not None:
            mix_shape = mix.shape
            mix = torch.stack([mix] * ensemble, dim=0)
            mix = mix.view((-1,) + mix_shape[1:])
            
            ### ADD 01 MAY (MINOR)
            if mask is not None:                           
                mask = (torch.stack([mask] * ensemble, dim=0)
                        .view((-1, mask.shape[-1])))
            ### ADD 01 MAY (MINOR)

        # pad to multiple of total downsampling to remove border effects
        mix_len = mix.shape[-1]
        mix, pad_ = self.pad(mix)
        if target is not None:
            target, _ = self.pad(target, pad=pad_)
        
        
        ### NEW 01 MAY
        if mask is not None:
            mask = torch.nn.functional.pad(
                mask, (pad_ // 2, pad_ - pad_ // 2))     # NEW – keep the same length
        ### NEW 01 MAY
        

        (mix, target), *denorm_args = self.normalize_batch((mix, target))
        mix_wav = mix
        mix = self.transform(mix)
        if target is not None:
            self.transform(target)
            
        # we set this up here to test the diffusion with a "perfect" score model
        if fake_score_snr is None: # we can test we some degraded score too
            score_snr = 5.0
        else:
            score_snr = fake_score_snr

        def score_wrapper(x, s, cond):
            if target is None:
                return self.score_model(x, s, cond)
            else:
                true_score = -(x - target) / s[:, None, None] ** 2
                score_rms = (true_score**2).mean().sqrt()
                noise_rms = score_rms * 10 ** (-score_snr / 20.0)
                noise = torch.randn(
                    true_score.shape,
                    dtype=true_score.dtype,
                    device=true_score.device,
                    generator=rng,
                )
                return true_score + noise * noise_rms

        # compute parameters
        delta_t = 1.0 / (n_steps - 1)
        gamma = (self.diff_kwargs.sigma_max / self.diff_kwargs.sigma_min) ** -delta_t
        eta = 1 - gamma**epsilon
        # beta = math.sqrt(1 - ((1 - eta) / gamma) ** 2)  # paper original
        beta = math.sqrt(1 - gamma ** (2 * (epsilon - 1.0)))  # in terms of gamma only

        # discretize time
        time = torch.linspace(0, 1, n_steps).type_as(mix)
        time = time.flip(dims=[0])
        sigma = self.get_std_dev(time)
        sigma = torch.broadcast_to(sigma[None, :], (mix.shape[0], sigma.shape[0]))

        # -------------- 
        # Condition
        # -------------- 
        # If text is provided, do new text path w/ debug prints; else old.
        if text is not None:
            # "new" approach
            # We might do debug logs:
            print(f"[DEBUG] 'enhance' sees text => using text in conditioner.")
            result = self.condition_model(mix, x_wav=mix_wav, text=text, train=True, mask = mask) # 01 May add mask
            if isinstance(result, tuple) and len(result) == 4:
                cond, aux_signal, aux_latent, _ = result  # ignoring text_metrics
            elif isinstance(result, tuple) and len(result) == 5:
                cond, aux_signal, aux_latent, _, _ = result  # ignoring text_metrics
            else:
                cond, aux_signal, aux_latent = result
        else:
            # old approach
            cond, aux_signal, aux_latent = self.condition_model(
                mix, x_wav=mix_wav, train=True, mask = mask) # 01 May add mask

        # x = None
        if use_aux_signal:
            x = self.aux_to_wav(aux_signal)
        else:
            # use diffusion

            # initial value
            if warm_start is None:
                x = randn(mix, sigma[:, 0], rng=rng)
                n_start = 0
            else:
                sig = self.aux_to_wav(aux_signal)
                x = sig + randn(sig, sigma[:, warm_start], rng=rng)
                n_start = warm_start

            for n in range(n_start, n_steps - 1):
                s_now = sigma[:, n]
                s_next = sigma[:, n + 1]
                score = score_wrapper(x, s_now, cond)
                z = randn(x, s_next, rng=rng)
                x = x + s_now[..., None, None] ** 2 * eta * score + beta * z

            # final step
            score = score_wrapper(x, sigma[:, -1], cond)
            x = x + sigma[:, -1, None, None] ** 2 * score

        # inverse transform
        x = self.transform(x, inv=True)

        # # remove the padding and restore signal scale
        # x = self.unpad(x, pad_)
        # x = torch.nn.functional.pad(x, (0, mix_len - x.shape[-1]))
        
        ### 01 MAY ADD
        # remove the padding and restore signal scale
        x = self.unpad(x, pad_)

        if mask is not None and pad_ > 0:                       # ← NEW
            mask = mask[..., pad_ // 2 : -(pad_ - pad_ // 2)]   # same crop

        x = torch.nn.functional.pad(x, (0, mix_len - x.shape[-1]))
        
         ### 01 MAY ADD
         
        
        if mask is not None:                       # keep tail at 0
            x = x * mask.unsqueeze(1)

        if keep_rms:
            x_rms = x.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-5)
            x = x * (mix_rms / x_rms)

        scale = x.abs().max(dim=-1, keepdim=True).values
        x = torch.where(scale > 1.0, x / scale, x)

        if ensemble is not None:
            x = x.view((-1,) + mix_shape)
            if ensemble_stat == "mean":
                x = x.mean(dim=0)
            elif ensemble_stat == "median":
                x = x.median(dim=0).values
            elif ensemble_stat == "signal_median":
                x = utils.signal_median(x)
            else:
                raise NotImplementedError()

        if x_ndim == 1:
            x = x[0, 0]
        elif x_ndim == 2:
            x = x[:, 0, :]
        return x

    def forward(self, xt, sigma, cond):
        return self.score_model(xt, sigma, cond)

    def get_std_dev(self, time):
        if self.diff_kwargs.schedule == "geometric":
            s_min = self.diff_kwargs.sigma_min
            s_max = self.diff_kwargs.sigma_max
            return s_min * (s_max / s_min) ** time
        else:
            raise NotImplementedError()
        
    
    def on_train_epoch_start(self):
        pass

    def adapt_time_sampling(self, x):
        with torch.no_grad():
            if not hasattr(self, "pr_cum"):
                # default to time uniform in first epoch
                time = x.new_zeros(x.shape[0]).uniform_()
            else:
                pr_cum = torch.broadcast_to(
                    self.pr_cum, (x.shape[0], self.pr_cum.shape[0])
                )
                time = x.new_zeros(x.shape[0])
                dice_roll = x.new_zeros(x.shape[0]).uniform_()
                for i in range(self.val_score_values.shape[0]):
                    ts, te = self.val_score_bins[i], self.val_score_bins[i + 1]
                    cand = x.new_zeros(x.shape[0]).uniform_() * (te - ts) + ts
                    time = torch.where(
                        torch.logical_and(
                            dice_roll >= pr_cum[:, i], dice_roll < pr_cum[:, i + 1]
                        ),
                        cand,
                        time,
                    )
        return time

    def sample_sigma(
        self, x, time_sampling="time_uniform", t_min=0.0, t_max=1.0, rng=None
    ):
        # sample the variance
        u = torch.rand(x.shape[0], generator=rng, dtype=x.dtype, device=x.device)
        time = (t_max - t_min) * u + t_min
        s_min = self.diff_kwargs.sigma_min
        s_max = self.diff_kwargs.sigma_max
        if time_sampling == "time_uniform":
            # geometric variance schedule
            sigma = self.get_std_dev(time)
        elif time_sampling == "sigma_linear":
            sigma = (s_max - s_min) * time + s_min
        elif time_sampling == "diffsym":
            # differential symmetric sampling
            # explanation:
            # 1) sample sigma uniformly
            # 2) apply a tranformation to time that has the same
            #    derivative as the standard deviation progression,
            #    but symmetric with respect to time
            # 3) then, apply the geometric progression
            sigma = (s_max - s_min) * time + s_min
            sigma = s_max + s_min - sigma
            num = torch.log10((s_max + s_min - sigma) / s_min)
            denom = math.log10(s_max / s_min)
            time = 1.0 - num / denom
            sigma = self.get_std_dev(time)
        elif time_sampling == "adaptive":
            time = self.adapt_time_sampling(x)
            sigma = self.get_std_dev(time)
        elif time_sampling == "time_discrete":
            n_steps = self.diff_kwargs.get("n_steps", 32)
            steps = torch.linspace(0.0, 1.0, n_steps).to(x.device)
            idx = abs(steps[:, None] - time[None, :]).min(dim=0).indices
            time = steps[idx]
            sigma = self.get_std_dev(time)
        elif time_sampling.startswith("time_normal"):
            try:
                alpha = torch.tensor(float(time_sampling.split("_")[2]))
            except (IndexError, ValueError):
                alpha = torch.tensor(
                    0.95
                )  # we want to use 100 * alpha % of the distribution

            time = utils.random.center_truncated_normal(
                area=alpha,
                min=t_min,
                max=t_max,
                size=x.shape[0],
                generator=rng,
                device=x.device,
            )
            sigma = self.get_std_dev(time)
        else:
            raise NotImplementedError()

        return sigma, time

    def compute_losses(
        self,
        mix,
        target,
        train=True,
        time_sampling="time_uniform",
        t_min=0.0,
        t_max=1.0,
        rng=None,
        text = None, ## NEW WITH TEXT ENCODER ###
        mask = None
    ):
        mix_trans = self.transform(mix)
        tgt_trans = self.transform(target)
        
        if mask is not None:
            m = mask.unsqueeze(1)
            mix     = mix * m
            target  = target * m
        # mix_trans  = self.transform(mix)      # REMOVED 03 MAY
        # tgt_trans  = self.transform(target)   # REMOVED 03 MAY
        

        if self.with_noise_target:
            noise = mix - target
            target_aux = torch.cat((target, noise), dim=1)
            target_aux_trans = torch.cat((tgt_trans, self.transform(noise)), dim=1)
        else:
            target_aux = target
            target_aux_trans = tgt_trans

        sigma, _ = self.sample_sigma(mix_trans, time_sampling, t_min, t_max, rng=rng)

        # sample the noise and create the target
        z = target.new_zeros(tgt_trans.shape).normal_(generator=rng)
        
        if mask is not None:
            z = z * m
        
        ### NEW ADD 19 APR ###
        z = target.new_zeros(tgt_trans.shape).normal_(generator=rng)
        z = _apply_mask(z, mask)             # keep score loss OK
        ### NEW ADD 19 APR ###
        
        x_t = tgt_trans + sigma[:, None, None] * z

        # run computations
        # cond, y_est, h_est = self.condition_model(mix_trans, x_wav=mix, train=True)
        
        if self.have_text:
            # THIS LINE CHANGES - Store the result which now includes text_metrics
            result = self.condition_model(mix_trans, x_wav=mix, text=text, train=True, mask = mask) ### 01 MAY: added mask
            if len(result) == 4:  # With text metrics
                cond, y_est, h_est, text_metrics = result
                # Store metrics for later use in training_step
                self.text_metrics = text_metrics
                
            elif len(result) == 5:  # With text metrics
                cond, y_est, h_est, text_metrics1, text_metrics2 = result
                # Store metrics for later use in training_step
                self.text_metrics1 = text_metrics1
                self.text_metrics2 = text_metrics2
                
            else:
                print(len(result))
                cond, y_est, h_est = result
                self.text_metrics = {}
        else:
            cond, y_est, h_est = self.condition_model(mix_trans, x_wav=mix, train=True, mask = mask) ### 01 MAY: added mask
            self.text_metrics = {}


        if self.detach_cond:
            cond = [c.detach() for c in cond]

        score = self.score_model(x_t, sigma, cond)

        # compute losses
        l_score = self.loss_score(sigma[..., None, None] * score, -z)
        if mask is not None:
            l_score = self.loss_score(sigma[...,None,None]*score*m, -z*m)
        else:
            l_score = self.loss_score(sigma[...,None,None]*score, -z)

        # if train:
        #     if self.losses_kwargs.weights.latent > 0.0 and h_est is not None:
        #         mel_target = self.condition_model.input_mel.compute_mel_spec(target_aux)
        #         mel_target = mel_target / torch.linalg.norm(
        #             mel_target, dim=(-2, -1), keepdim=True
        #         ).clamp(min=1e-5)
        #         l_latent = self.loss_latent(h_est, mel_target)

        ### NEW CHANGE 19 APR ###
        if train:
            if self.losses_kwargs.weights.latent > 0.0 and h_est is not None:
                mel_target = self.condition_model.input_mel.compute_mel_spec(
                    _apply_mask(target_aux, mask)
                )                
                
                mel_target = mel_target / torch.linalg.norm(
                    mel_target, dim=(-2, -1), keepdim=True
                ).clamp(min=1e-5)
                l_latent = self.loss_latent(h_est, mel_target)
                
                # down‑sample the mask to mel length once
                mel_mask = _downsample_mask(mask, mel_target.shape[-1] * target.shape[-1] // mel_target.shape[-1])
                h_est_m   = _apply_mask(h_est, mel_mask.squeeze(1))
                mel_target= _apply_mask(mel_target, mel_mask.squeeze(1))
                l_latent  = self.loss_latent(h_est_m, mel_target)
        ### NEW CHANGE 19 APR ###
        
            else:
                l_latent = l_score.new_zeros(1)

            if self.losses_kwargs.weights.signal > 0.0:
                l_signal = self.loss_signal(y_est, target_aux_trans)
            
            ### NEW CHANGE 19 APR ###
            if self.losses_kwargs.weights.signal > 0.0:
                y_est_m  = _apply_mask(y_est,  mask)
                tgt_m    = _apply_mask(target_aux_trans, mask)
                l_signal = self.loss_signal(y_est_m, tgt_m)            
            ### NEW CHANGE 19 APR ###
            
            else:
                l_signal = l_score.new_zeros(1)

            loss = self.losses_kwargs.weights.score * l_score
            if torch.isnan(l_score):
                log.warn("Score loss is nan...")
                breakpoint()

            if not torch.isnan(l_signal):
                loss = loss + self.losses_kwargs.weights.signal * l_signal
            else:
                log.warn("Signal loss is nan, skip for total loss")

            if not torch.isnan(l_latent):
                loss = loss + self.losses_kwargs.weights.latent * l_latent
            else:
                log.warn("Latent loss is nan, skip for total loss")

            return loss, l_score, l_signal, l_latent
        else:
            return l_score

    # -------------------------------------------------------------------
    #   TRAINING & VALIDATION - minimal text additions 
    # -------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        If batch has 3 items => text is present => new approach.
        Else => old approach exactly.
        """
        # if len(batch) >= 3 and isinstance(batch[2], (str, list, torch.Tensor)):
        #     self.have_text = True
        #     mix, target, text = batch[:3]
        #     print(f"[DEBUG] training_step sees text => {text}")
        
        if len(batch) == 4:                     # mix, tgt, text?, mask
            mix, target, text, mask = batch
            self.have_text = isinstance(text, (str, list, torch.Tensor))
        elif len(batch) == 3 and isinstance(batch[2], (str, list, torch.Tensor)):
            self.have_text = True
            mix, target, text = batch[:3]
            mask = None
        
        else:
            self.have_text = False
            mix, target = batch[:2]
            mask = None
              

        if getattr(self.train_kwargs, "dynamic_mixing", False):
            noise = mix - target
            perm = torch.randperm(noise.shape[0])
            mix = target + noise[perm, ...]

        (mix, target), *stats = self.normalize_batch(
            (mix, target), norm=self.normalization_norm
        )
        
        
        if mask is not None:                    # mask padded frames
            m = mask.unsqueeze(1)
            mix    = mix * m
            target = target * m

        # We unify old vs. new:
        if self.have_text:
            # pass text into compute_losses
            loss, l_score, l_signal, l_latent = self.compute_losses(
                mix,
                target,
                train=True,
                time_sampling=self.train_kwargs.time_sampling,
                text=text,  # ADDED
                mask=mask   # ← NEW
            )
            
        else:
            # old approach
            loss, l_score, l_signal, l_latent = self.compute_losses(
                mix,
                target,
                train=True,
                time_sampling=self.train_kwargs.time_sampling,
            )

        # every 10 steps, we log stuff
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            batch_size=mix.shape[0],
        )
        kwargs = dict(
            batch_size=mix.shape[0],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.log("train/score", l_score, **kwargs)
        self.log("train/signal", l_signal, **kwargs)
        self.log("train/latent", l_latent, **kwargs)

        self.do_lr_warmup()

        return loss
    

    # -------------------------------------------------------------------
    #   The rest of old code: validation, configure_optimizers, ...
    # -------------------------------------------------------------------
    def on_train_epoch_start(self):
        pass

    def on_validation_epoch_start(self):
        self.n_batches_est_done = 0
        self.n_tb_samples_saved = 0
        self.num_tb_samples = self.val_kwargs.get("num_tb_samples", 5)
        if not hasattr(self, "first_val_done"):
            self.first_val_done = False
        else:
            self.first_val_done = True

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(682479040)

    def validation_step(self, batch, batch_idx, dataset_i=0):
        """
        If text is present in batch => use new path, else old path.
        No difference to the old code for no-text scenario.
        """
        # detect text
        # if len(batch) >= 3 and isinstance(batch[2], (str, list, torch.Tensor)):
        #     self.have_text = True
        #     mix_raw, target_raw, text = batch[:3]
        # else: # same as in original
        #     self.have_text = False
        #     mix_raw, target_raw = batch[:2]
            
        # ------------------- unpack ---------------------
        if len(batch) == 4:                       # mix, tgt, txt?, mask
            mix_raw, target_raw, text, mask = batch
            self.have_text = isinstance(text, (str, list, torch.Tensor))
        elif len(batch) == 3 and isinstance(batch[2], (str, list, torch.Tensor)):
            mix_raw, target_raw, text = batch
            mask = None
            self.have_text = True
        else:                                     # legacy 2‑tuple
            mix_raw, target_raw = batch[:2]
            text, mask = None, None
            self.have_text = False

        # batch_scaled, *stats = self.normalize_batch((mix_raw, target_raw), norm=self.normalization_norm)
        # mix, target = batch_scaled
        
        # ------------- apply mask BEFORE normalisation -----------
        if mask is not None:
            m = mask.unsqueeze(1)                 # [B,1,T]
            mix_raw    = mix_raw    * m
            target_raw = target_raw * m

        batch_scaled, *stats = self.normalize_batch(
            (mix_raw, target_raw), norm=self.normalization_norm
        )
        mix, target = batch_scaled
        
        
        batch_size = mix.shape[0]

        tb = torch.linspace(0.0, 1.0, self.val_kwargs.n_bins + 1, device=mix.device)
        bin_scores = []
        for i in range(self.val_kwargs.n_bins):
            # ----------- pad mix, target *and* mask consistently ----------
            mix_p,   pad_ = self.pad(mix)
            target_p, _   = self.pad(target, pad=pad_)
            if mask is not None:
                mask_p = torch.nn.functional.pad(
                    mask, (pad_ // 2, pad_ - pad_ // 2)
                )
            else:
                mask_p = None

            if self.have_text:
                ls = self.compute_losses(
                    mix_p[0],
                    target_p[0],
                    train=False,
                    time_sampling="time_uniform", # always sample uniformly for validation
                    t_min=tb[i],
                    t_max=tb[i + 1],
                    rng=self.rng,
                    text=text, ## NEW WITH TEXT ENCODER ###
                    mask=mask_p,
                )
            else: # same as in original
                ls = self.compute_losses(
                    mix_p[0],
                    target_p[0],
                    train=False,
                    time_sampling="time_uniform", # always sample uniformly for validation
                    t_min=tb[i],
                    t_max=tb[i + 1],
                    rng=self.rng,
                    mask=mask_p,
                )
            bin_scores.append(ls)

        self.val_score_bins = tb
        self.val_score_values = torch.tensor(bin_scores, device=mix.device)
        l_score = torch.mean(self.val_score_values)
        
        # compute the cumulative distribution
        # manual cumsum to be deterministic
        v = self.val_score_values.clamp(min=5e-4)
        pr_cum = v.new_zeros(v.shape[0] + 1)
        for idx, p in enumerate(v):
            pr_cum[idx + 1] = pr_cum[idx] + p
        pr_cum = pr_cum / pr_cum[-1]
        pr_cum[-1] = 1.0 + 1e-5 # to include the last bound
        self.pr_cum = pr_cum

        self.log(
            "val/score", l_score, on_epoch=True, sync_dist=True, batch_size=batch_size
        )
        for i in range(self.val_kwargs.n_bins):
            self.log(
                f"val/score_{tb[i]:.2f}-{tb[i+1]:.2f}",
                bin_scores[i],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        # Validation enhancement losses
        if self.trainer.testing or self.n_batches_est_done < self.val_kwargs.max_enh_batches:
            self.n_batches_est_done += 1
            # use unnormalized data for enhancement
            # if self.have_text:
            #     mix_, target_, text_ = batch[:3] 
            #     # mix_, target_, text_ = batch # to check if it works (in line with original for non-text)
            #     print(f"[VALIDATION DEBUG] Before enhance call, text available: {text is not None}")
            #     print(f"[VALIDATION DEBUG] Text sample: {text[0] if text is not None else 'None'}")
            #     est = self.enhance(mix_, rng=self.rng, text=text_)
            # else:
            #     # mix_, target_ = batch[:2]
            #     mix_, target_ = batch # to check if it works (per original)
            #     print(f"[VALIDATION DEBUG] Before enhance call, not using text")
            #     est = self.enhance(mix_, rng=self.rng)
            
            # ------------- enhancement with mask --------------
            if self.have_text:
                mix_, target_, text_, mask_ = mix_raw, target_raw, text, mask
                est = self.enhance(mix_, rng=self.rng, text=text_, mask=mask_)
            else:
                mix_, target_, mask_ = mix_raw, target_raw, mask
                est = self.enhance(mix_, rng=self.rng, mask=mask_)

            # zero‑out padding region in the estimate so metrics ignore it
            if mask_ is not None:
                est = est * mask_.unsqueeze(1)
                target_ = target_ * mask_.unsqueeze(1)
            
            # Log validation text metrics
            if self.have_text and hasattr(self, 'text_metrics') and self.text_metrics:
                for k, v in self.text_metrics.items():
                    if isinstance(v, (int, float)):
                        self.log(f"val_text_checks/{k}", v, on_epoch=True, sync_dist=True, batch_size=batch_size)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"val_text_checks/top_attended_{i}", pos, on_epoch=True, sync_dist=True, batch_size=batch_size)
            
            if self.have_text and hasattr(self, 'text_metrics1') and self.text_metrics1:
                for k, v in self.text_metrics1.items():
                    if isinstance(v, (int, float)):
                        self.log(f"val_text_checks1/{k}", v, on_epoch=True, sync_dist=True, batch_size=batch_size)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"val_text_checks1/top_attended_{i}", pos, on_epoch=True, sync_dist=True, batch_size=batch_size)
            
            if self.have_text and hasattr(self, 'text_metrics2') and self.text_metrics2:
                for k, v in self.text_metrics2.items():
                    if isinstance(v, (int, float)):
                        self.log(f"val_text_checks2/{k}", v, on_epoch=True, sync_dist=True, batch_size=batch_size)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"val_text_checks2/top_attended_{i}", pos, on_epoch=True, sync_dist=True, batch_size=batch_size)

            # log val losses
            for name, loss in self.enh_losses.items():
                val_metric = loss(est, target_)
                if not isinstance(val_metric, dict):
                    val_metric = {"": val_metric}
                for sub_name, loss_val in val_metric.items():
                    self.log(
                        f"{name}{sub_name}",
                        loss_val.to(mix_.device),
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=batch_size,
                    )

            # optional code: wandb logs etc. 
            # (unchanged from old code, just do the normal audio logging)
            
            # ✅ Save a few audio samples to Wandb (only in main process)
            if self.trainer.is_global_zero and self.n_tb_samples_saved < self.num_tb_samples:
                num_save = min(self.num_tb_samples - self.n_tb_samples_saved, batch_size)
                audio_logs = {}

                print(f"[DEBUG] Logging {num_save} audio samples to Wandb...")

                for idx in range(num_save):
                    sample_id = f"sample_{self.global_rank}_{self.n_tb_samples_saved}"

                    # Debug print for sample info
                    print(f"[DEBUG] Processing sample {sample_id}")

                    # Normalize for logging
                    mix_ = mix[idx] * 0.95 / torch.max(torch.abs(mix[idx]))
                    mix_loud = torchaudio.functional.loudness(mix[idx], self.fs)

                    # ✅ Convert tensor to NumPy and ensure correct dtype & shape
                    mix_np = mix_.cpu().numpy().astype(np.float32)

                    # Ensure the shape is (samples,)
                    if mix_np.ndim == 2:
                        mix_np = mix_np.reshape(-1)  # Convert from (1, samples) to (samples,)

                    # print(f"[DEBUG] Input audio shape: {mix_np.shape}, dtype: {mix_np.dtype}, max: {mix_np.max()}, min: {mix_np.min()}")

                    # Check for NaNs or Infs before logging
                    if np.isnan(mix_np).any() or np.isinf(mix_np).any():
                        print(f"[ERROR] Detected NaN or Inf values in input audio: {sample_id}")
                        continue  # Skip this sample

                    # Ensure a standard sample rate (e.g., 16kHz)
                    sample_rate = 16000  # Change if needed

                    # Log input audio
                    audio_logs[f"audio/input_{sample_id}"] = wandb.Audio(
                        mix_np, sample_rate=sample_rate, caption="Noisy Input"
                    )

                    if not self.first_val_done:
                        # Save clean target the first time
                        tgt_loud = torchaudio.functional.loudness(target[idx], self.fs)
                        tgt_gain = 10 ** ((mix_loud - tgt_loud) / 20)

                        target_np = (target[idx] * tgt_gain).cpu().numpy().astype(np.float32)
                        if target_np.ndim == 2:
                            target_np = target_np.reshape(-1)

                        print(f"[DEBUG] Target audio shape: {target_np.shape}, dtype: {target_np.dtype}, max: {target_np.max()}, min: {target_np.min()}")

                        if np.isnan(target_np).any() or np.isinf(target_np).any():
                            print(f"[ERROR] Detected NaN or Inf values in target audio: {sample_id}")
                            continue  # Skip this sample

                        audio_logs[f"audio/target_{sample_id}"] = wandb.Audio(
                            target_np, sample_rate=sample_rate, caption="Clean Target"
                        )

                    # Log enhanced output
                    est_loud = torchaudio.functional.loudness(est[idx], self.fs)
                    est_gain = 10 ** ((mix_loud - est_loud) / 20)

                    est_np = (est[idx] * est_gain).cpu().numpy().astype(np.float32)
                    if est_np.ndim == 2:
                        est_np = est_np.reshape(-1)

                    # print(f"[DEBUG] Output audio shape: {est_np.shape}, dtype: {est_np.dtype}, max: {est_np.max()}, min: {est_np.min()}")

                    if np.isnan(est_np).any() or np.isinf(est_np).any():
                        print(f"[ERROR] Detected NaN or Inf values in output audio: {sample_id}")
                        continue  # Skip this sample

                    audio_logs[f"audio/output_{sample_id}"] = wandb.Audio(
                        est_np, sample_rate=sample_rate, caption="Enhanced Output"
                    )

                    # Update sample count
                    self.n_tb_samples_saved += 1
                    if self.n_tb_samples_saved >= self.num_tb_samples:
                        print(f"[DEBUG] Reached num_tb_samples limit ({self.num_tb_samples}), stopping logging.")
                        break

                # ✅ Log all samples at once (efficient logging)
                if audio_logs:
                    print("[DEBUG] Sending audio logs to Wandb...")
                    self.logger.experiment.log(audio_logs, step=self.global_step)
                else:
                    print("[DEBUG] No audio logs found! Wandb log skipped.")


    def on_validation_epoch_end(self):
        print(f"[DEBUG] ✅ Validation epoch complete")

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataset_i=None):
        return self.validation_step(batch, batch_idx, dataset_i=dataset_i)

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        opt_kwargs = OmegaConf.to_container(self.opt_kwargs, resolve=True)
        
        # the warm-up parameter is handled separately
        self.lr_warmup = opt_kwargs.pop("lr_warmup", None)
        self.lr_original = self.opt_kwargs.lr

        # We can have a list of keywords to exclude from weight decay
        weight_decay = opt_kwargs.pop("weight_decay", 0.0)
        wd_exclude_list = opt_kwargs.pop("weight_decay_exclude", [])

        def pick_excluded(name):
            return any([kw in name for kw in wd_exclude_list])

        # excluded = []
        # others = []
        # for (name, p) in self.named_parameters():
        #     if not p.requires_grad:
        #         continue
        #     if pick_excluded(name):
        #         excluded.append(p)
        #     else:
        #         others.append(p)

        # without_weight_decay = {"params": excluded}
        # with_weight_decay = {"params": others, "weight_decay": weight_decay}

        # log.info(f"set optim with {self.opt_kwargs}")
        # opt_kwargs = {**{"params": [without_weight_decay, with_weight_decay]}, **opt_kwargs}
        # optimizer = instantiate(config=opt_kwargs, _recursive_=False, _convert_="all")
        
        
        excluded = [
            p
            for (name, p) in self.named_parameters()
            if pick_excluded(name) and p.requires_grad
        ]
        others = [
            p
            for (name, p) in self.named_parameters()
            if not pick_excluded(name) and p.requires_grad
        ]

        without_weight_decay = {"params": excluded}
        with_weight_decay = {"params": others, "weight_decay": weight_decay}

        # we may have some frozen layers, so we remove these parameters
        # from the optimization
        log.info(f"set optim with {self.opt_kwargs}")
        opt_kwargs = {
            **{"params": [without_weight_decay, with_weight_decay]},
            **opt_kwargs,
        }
        optimizer = instantiate(config=opt_kwargs, _recursive_=False, _convert_="all")

        if self.schedule_kwargs is not None:
            if "scheduler" not in self.schedule_kwargs:
                scheduler = instantiate(
                    {**self.schedule_kwargs, **{"optimizer": optimizer}}
                )
            else:
                scheduler = OmegaConf.to_container(self.schedule_kwargs, resolve=True)
                lr_sch_kwargs = scheduler.pop("scheduler")
                scheduler["scheduler"] = instantiate(
                    {**lr_sch_kwargs, **{"optimizer": optimizer}}, _recursive_=False
                )
        else:
            scheduler = None

        # this will be called in on_after_backward
        self.grad_clipper = instantiate(self.grad_clip_kwargs)

        if scheduler is None:
            return [optimizer]
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.val_kwargs.main_loss,
            }

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update(self.model_parameters())

    def on_after_backward(self):
        if self.grad_clipper is not None:
            grad_norm, clipping_threshold = self.grad_clipper(self)
        else:
            # we still want to compute this for monitoring in tensorboard
            grad_norm = utils.grad_norm(self)
            clipping_threshold = grad_norm

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)
            
            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]
            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )

    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if self.ema is not None:
            if ema is not None:
                self.ema.load_state_dict(ema)
            else:
                self._error_loading_ema = True
                log.warn("EMA state_dict not found in checkpoint!")

    def train(self, mode=True, no_ema=False):
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode

        if self.ema is None:
            return res

        if not self._error_loading_ema:
            if mode is False and not no_ema:
                # eval
                self.ema.store(self.model_parameters())  # store current params in EMA
                self.ema.copy_to(
                    self.model_parameters()
                )  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(
                        self.model_parameters()
                    )  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def to(self, *args, **kwargs):
        if self.ema is not None:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def do_lr_warmup(self):
        if not hasattr(self, "lr_warmup"):
            return
        if self.lr_warmup is not None and self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            optimizer = self.trainer.optimizers[0]
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr_original
