# universe_gan_NS_11May_tg.py 
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
# from .m_universe_NS_adj3_fix_map_orig import Universe 
# from .universe_NS_04May import Universe 
# from .universe_NS_10May_fix import Universe 
from .universe_NS_10May_fix_tg import Universe 

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
    
   
    
    
    ###
    
    ## UPD 6 MAY v2#
    @staticmethod
    def _guided_attn_loss(attn, q_mask, s_mask, sigma=0.15):
        """
        Guided-attention loss   (lower = better diagonal)

        attn   : [B, Q, S_total]   soft-max weights (blanks + pad kept)
        q_mask : [B, Q]            True  = valid mel frame
        s_mask : [B, S_total]      True  = valid token  (blanks **also** True here)

        We:
        • build a diagonal guide with slope  S_total / Q
        • zero-out guide on blanks & pads  (so they’re not penalised)
        • normalise loss by #valid cells and by min(Q,S) to remove length bias
        """
        B, Q, S_total = attn.shape
        device = attn.device
        
        
         # ---------- diagonal guide (adaptive slope) -----------------------
        ratio = S_total / (Q + 1e-5)                            # text-to-mel rate
        q_pos = torch.linspace(0.0, 1.0, Q,       device=device).view(1, Q, 1)  # [1,Q,1]
        s_pos = torch.linspace(0.0, 1.0, S_total, device=device).view(1, 1, S_total)

    


        ## UPD 08 MAY ##
        # Use multiple Gaussians with different widths for more robust guidance
        sigma_narrow = (sigma * 0.5 * torch.sqrt(
                s_mask.sum(1, keepdim=True).float() /
                (q_mask.sum(1, keepdim=True).float() + 1e-5)
            ).clamp_min(0.03)).unsqueeze(-1)
            
        sigma_wide = (sigma * 2.0 * torch.sqrt(
                s_mask.sum(1, keepdim=True).float() /
                (q_mask.sum(1, keepdim=True).float() + 1e-5)
            ).clamp_min(0.1)).unsqueeze(-1)
            
        # Multi-scale guidance (narrow + wide)
        guide_narrow = 1.0 - torch.exp(-((q_pos - s_pos / ratio) ** 2) / (2 * sigma_narrow ** 2))
        guide_wide = 0.5 * (1.0 - torch.exp(-((q_pos - s_pos / ratio) ** 2) / (2 * sigma_wide ** 2)))
        guide = guide_narrow + guide_wide  # Combine both guidance signals
        ## UPD 08 MAY ##
        


        # ---------- remove blanks + pad from the penalty ------------------
        blank_or_pad = ~s_mask                                    # True on blanks & pads
        guide = guide.masked_fill(blank_or_pad.unsqueeze(1), 0.0) # 0-weight penalty

        # ---------- masks for real cells ----------------------------------
        pad = (~q_mask).unsqueeze(-1) | (~s_mask).unsqueeze(1)     # invalid cells
        valid = ~pad

        # ---------- per-example, length-normalised GA ---------------------
        per_ex = (
            (attn * guide).masked_fill(pad, 0.0).sum((-1, -2)) /
            valid.float().sum((-1, -2)).clamp_min(1.0)
        )

        # further divide by min(Q,S) to flatten curriculum spikes
        seq_len_norm = torch.minimum(q_mask.sum(1), s_mask.sum(1)).float().clamp_min(1.0)
        per_ex = per_ex / seq_len_norm

        # final batch mean
        return per_ex.mean()
    ## UPD 6 MAY v2#
    
    

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
        tg = None # NEW 15 MAY

            
        
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
        
        ### NEW 15 MAY ###
        if len(batch) == 5:                          # mix, tgt, text, mask, tg  ← NEW
            mix_raw, target_raw, text, mask, tg = batch
            text_str_list = text if isinstance(text, list) else [text]
            target_original = target_raw
    
        # if len(batch) == 4:                          # ← NEW
        elif len(batch) == 4:                          # ← NEW
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
       
            
            # ——————————————————————————————————————————
            # NEW Return-layout: (cond, y_est, h, text_metrics)
            # text_metrics = {"mel": {…}, "lat": {…}}
            # ——————————————————————————————————————————
            
            ### ADD 15 MAY ###
            if tg:
                cond, y_est, _, text_metrics = self.condition_model(
                    mix, text=text_str_list, train=True, mask=mask, tg = tg # NEW 15 MAY - incl. tg
                )
            ### ADD 15 MAY ###
            else:
                cond, y_est, _, text_metrics = self.condition_model(
                    mix, text=text_str_list, train=True, mask=mask
                )
                
            def _log_metrics(level: str, metrics: dict):
                for k, v in metrics.items():
                    tag = f"text_{level}/{k}"
                    if isinstance(v, (int, float)):
                        self.log(tag, v, batch_size=mix.shape[0], **self.log_kwargs)
                    elif isinstance(v, list) and k == "top_attended_positions" and len(v) <= 5:
                        for i, pos in enumerate(v):
                            self.log(f"{tag}_{i}", pos, batch_size=mix.shape[0], **self.log_kwargs)

            if isinstance(text_metrics, dict):
                if "mel" in text_metrics:
                    _log_metrics("mel", text_metrics["mel"])
                if "lat" in text_metrics:
                    _log_metrics("lat", text_metrics["lat"])
            # ------------------------------------------------------------------
            
            ### UPD 10 MAY ###
            
            
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
        
        

                
        #### UPD 05 MAY ####
        # ===== Guided‑attention loss (NEW) =========================
        ga_w = self.losses_kwargs.weights.get("guided_attn", 0.0)
        # print(f"[DEBUG] Guided attention loss weight: {ga_w=}")
        # if ga_w > 0.0 and use_text:
        #     # tc = self.condition_model.text_conditioner
        #     tc = self.condition_model.text_cond_mel
        
        if ga_w > 0.0 and use_text:
            # pick whichever level actually carries cross-attention
            tc = None
            for cand in (self.condition_model.text_cond_mel,
                         self.condition_model.text_cond_lat):
                if cand is not None and hasattr(cand, "last_attn_map"):
                    tc = cand; break        
        
            attn_map = getattr(tc, "last_attn_map", None)
            q_mask   = getattr(tc, "last_q_mask",   None)
            s_mask   = getattr(tc, "last_s_mask",   None)
            if None not in (attn_map, q_mask, s_mask):
                loss_ga = self._guided_attn_loss(attn_map, q_mask, s_mask, sigma=0.15)
                
                #### 06 MAY - LINEAR ANNEALING OF GA WEIGHT AND COV LOSS

                # linear anneal of GA weight
                steps  = self.losses_kwargs.weights.get("ga_anneal_steps", 30000)
                w_ga   = ga_w * max(0.0, 1.0 - self.global_step / steps)
                loss_gen = loss_gen + w_ga * loss_ga

                
                ## UPD 6 MAY v2#
                # -------  ▼  replace coverage computation ▼  -------------------------
                token_cov = (attn_map.sum(1) - 1.0).abs()   # [B, S_total]
                token_cov = token_cov.masked_select(s_mask) # keep real text tokens only
                cov       = token_cov.mean()
                # ---------------------------------------------------------------------
                ## UPD 6 MAY v2#
                
                
                w_cov = self.losses_kwargs.weights.get("coverage", 0.1)
                loss_gen = loss_gen + w_cov * cov
                #### 06 MAY - LINEAR ANNEALING OF GA WEIGHT AND COV LOSS



                ## UPD 8 MAY ##
                # Add token concentration penalty
                # Calculate per-token attention (averaged over frames)
                token_attention = attn_map.masked_fill(~s_mask.unsqueeze(1), 0.0).sum(1)  # [B, S]
                token_attention = token_attention / token_attention.sum(-1, keepdim=True).clamp_min(1e-8)
                
                # Penalize tokens that get too much attention (entropy-based)
                valid_tokens = s_mask.sum(-1).clamp_min(1.0)  # count of valid tokens per batch
                token_entropy = -(token_attention * torch.log(token_attention.clamp_min(1e-8))).sum(-1) / torch.log(valid_tokens)
                token_conc_penalty = (1.0 - token_entropy).mean()  # penalize low entropy (high concentration)
                
                w_conc = self.losses_kwargs.weights.get("token_concentration", 0.5)
                loss_gen = loss_gen + w_conc * token_conc_penalty
                ## UPD 8 MAY ##


                
                self.log("train/guided_attn", loss_ga, **self.log_kwargs)
                self.log("train/cover_loss", cov, **self.log_kwargs)
                self.log("train/token_concentration", token_conc_penalty, **self.log_kwargs) # UPD 8 MAY
                # loss_gen = loss_gen + ga_w * loss_ga
                print(f"[DEBUG] Guided attention loss value: {loss_ga.item()=}")
            else:
                print(f"[DEBUG] Guided attention loss: missing tensors in {tc=}")
        # ==========================================================
        #### UPD 05 MAY ####
        
        
        
        #### 15 MAY - ADD MFA LOSS ####
        # ───────────────────────────────────────────────────────────
        # NEW ❶ Phoneme-position alignment loss
        # ───────────────────────────────────────────────────────────
        align_w = self.losses_kwargs.weights.get("align_phoneme", 0.0)
        if align_w > 0.0 and use_text:
            tc = None
            for cand in (self.condition_model.text_cond_mel,
                         self.condition_model.text_cond_lat):
                if cand is not None and getattr(cand, "last_coords", None) is not None:
                    tc = cand; break

            if tc is not None:
                attn_map  = tc.last_attn_map          # [B,Q,S]
                coords_b  = tc.last_coords            # List[Tensor [S,2]]
                if attn_map is not None and coords_b:
                    pred_pos = attn_map.argmax(dim=1).float()  # [B,S]
                    
                    losses_align = []
                    
                    ### TO FIX MULTI-GPU ISSUE ###
                    local_B = pred_pos.size(0)
                    global_offset = self.module_offset if hasattr(self, "module_offset") else 0
                    coords_b = coords_b[global_offset : global_offset + local_B]
                    ### TO FIX MULTI-GPU ISSUE ###


                    for b, (coords, pred) in enumerate(zip(coords_b, pred_pos)):
                        # skip if no spans or all zeros
                        if not coords.numel() or (coords == 0).all():
                            print(f"[DEBUG] skip align for item {b}: no valid coords")
                            continue

                        centers  = coords.float().mean(dim=1)   # per-token centre frames
                        seq_len  = pred.size(0)                 # length of this sample
                        L        = min(centers.size(0), seq_len)

                        if centers.size(0) > seq_len:
                            print(f"[DEBUG ALIGN] truncating coords {centers.size(0)} -> {seq_len}")

                        diff = torch.relu((pred[:L] - centers[:L]).abs() - 2.0)  # 2-frame slack
                        losses_align.append(diff.mean())
                    
                        
                            
                    if losses_align:
                        loss_align = torch.stack(losses_align).mean()
                        loss_gen   = loss_gen + align_w * loss_align
                        self.log("train/align_phoneme", loss_align,
                                 **self.log_kwargs)
                        
                    # for b, coords in enumerate(coords_b):
                    #     if coords.numel():
                    #         # compute per-token centre positions
                    #         centers = coords.float().mean(dim=1)  # [n_coords]
                    #         max_frames = pred_pos.size(1)
                    #         n = centers.size(0)
                    #         # truncate if more tokens than frames
                    #         if n > max_frames:
                    #             print(f"[DEBUG ALIGN] Truncating {n}->{max_frames}")
                    #             centers = centers[:max_frames]
                    #             n = max_frames
                    #         # take top-attention frame for each token
                    #         pred = pred_pos[b, :n]
                    #         diff = (pred - centers).abs()
                    #         diff = torch.relu(diff - 2.0)         # allow 2-frame slack
                    #         losses_align.append(diff.mean())    
                    
                    
                    
        #### 15 MAY - ADD MFA LOSS ####
      
        
        ## UPD 6 MAY v2#
        div_w = self.losses_kwargs.weights.get("head_div", 0.0)
        if div_w > 0.0 and use_text:
            # div_loss = getattr(self.condition_model
            #                 .text_conditioner
            #                 .cross_attention, "div_loss", None)
            div_loss = getattr(self.condition_model
                            .text_cond_mel
                            .cross_attention, "div_loss", None)
            if div_loss is not None:
                loss_gen = loss_gen + div_w * div_loss
                self.log("train/head_div", div_loss, **self.log_kwargs)
        ## UPD 6 MAY v2#
                
        # 6) MPD gen    

        
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

        # use_text_branch = getattr(self.condition_model, "text_conditioner", None) is not None
        use_text_branch = getattr(self.condition_model, "text_cond_mel", None) is not None

        text_params = []
        # if use_text_branch:
        #     text_params = [p for n, p in self.condition_model.named_parameters()
        #                    if n.startswith("text_conditioner") and p.requires_grad]
            
        if use_text_branch:
            text_params = [p for n, p in self.condition_model.named_parameters()
                           if n.startswith("text_cond") and p.requires_grad]    
            
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