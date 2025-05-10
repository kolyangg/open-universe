# condition_NS_plbert_adj_clean_ce_check
# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
Conditioner network for the UNIVERSE model, merging old condition.py with new text logic.
- If no text is provided, the code behaves exactly like the old condition.py.
- If text is present, we do FiLM + cross-attention (PL-BERT or similar) plus debug logs.
"""

import math
import torch
import torchaudio
from hydra.utils import instantiate

from .blocks import (
    BinomialAntiAlias,
    ConvBlock,
    PReLU_Conv,
    cond_weight_norm,
)

import os
import datetime
from collections import defaultdict


def make_st_convs(
    ds_factors,
    input_channels,
    num_layers=None,
    use_weight_norm=False,
    use_antialiasing=False,
):
    if num_layers is None:
        num_layers = len(ds_factors) - 1
    st_convs = torch.nn.ModuleList()
    rates = [ds_factors[-1]]
    for r in ds_factors[-2::-1]:
        rates.append(rates[-1] * r)
    rates = rates[::-1]
    for i in range(len(ds_factors)):
        if i >= num_layers:
            st_convs.append(None)
        else:
            i_chan = input_channels * 2**i
            o_chan = input_channels * 2 ** len(ds_factors)
            new_block = PReLU_Conv(
                i_chan,
                o_chan,
                kernel_size=rates[i],
                stride=rates[i],
                use_weight_norm=use_weight_norm,
            )
            if use_antialiasing:
                new_block = torch.nn.Sequential(
                    BinomialAntiAlias(rates[i] * 2 + 1), new_block
                )
            st_convs.append(new_block)
    return st_convs


class MelAdapter(torch.nn.Module):
    def __init__(
        self, n_mels, output_channels, ds_factor, oversample=2, use_weight_norm=False
    ):
        super().__init__()
        self.ds_factor = ds_factor
        n_fft = oversample * ds_factor
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=ds_factor,
            center=False,
        )
        self.conv = cond_weight_norm(
            torch.nn.Conv1d(n_mels, output_channels, kernel_size=3, padding="same"),
            use=use_weight_norm,
        )
        self.conv_block = ConvBlock(output_channels, use_weight_norm=use_weight_norm)

        # workout the padding to get a good number of frames
        pad_tot = n_fft - ds_factor
        self.pad_left, self.pad_right = pad_tot // 2, pad_tot - pad_tot // 2

    def compute_mel_spec(self, x):
        r = x.shape[-1] % self.ds_factor
        if r != 0:
            pad = self.ds_factor - r
        else:
            pad = 0
        x = torch.nn.functional.pad(x, (self.pad_left, pad + self.pad_right))
        x = self.mel_spec(x)  # => [B, n_mels, T_mel] (plus the old single freq dimension)
        x = x.squeeze(1)      # remove channel dim

        # the paper mentions only that they normalize the mel-spec, not how
        # I am trying a simple global normalization so that frames have
        # unit energy on average
        norm = (x**2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).sqrt()
        x = x / norm.clamp(min=1e-5)

        return x

    def forward(self, x):
        x = self.compute_mel_spec(x)
        x = self.conv(x)
        x, *_ = self.conv_block(x)
        return x


class ConditionerEncoder(torch.nn.Module):
    def __init__(
        self,
        ds_factors,
        input_channels,
        with_gru_residual=False,
        with_extra_conv_block=False,
        act_type="prelu",
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
    ):
        super().__init__()

        self.with_gru_residual = with_gru_residual
        self.extra_conv_block = with_extra_conv_block

        c = input_channels

        self.ds_modules = torch.nn.ModuleList(
            [
                ConvBlock(
                    c * 2**i,
                    r,
                    "down",
                    act_type=act_type,
                    use_weight_norm=use_weight_norm,
                    antialiasing=use_antialiasing,
                )
                for i, r in enumerate(ds_factors)
            ]
        )

        # the strided convolutions to adjust rate and channels to latent space
        self.st_convs = make_st_convs(
            ds_factors,
            input_channels,
            num_layers=len(ds_factors) - 1,
            use_weight_norm=use_weight_norm,
            use_antialiasing=use_antialiasing,
        )

        if self.extra_conv_block:
            self.ds_modules.append(
                ConvBlock(
                    c * 2 ** len(ds_factors),
                    act_type=act_type,
                    use_weight_norm=use_weight_norm,
                )
            )
            self.st_convs.append(None)

        oc = input_channels * 2 ** len(ds_factors)  # number of output channels

        self.seq_model = seq_model
        if seq_model == "gru":
            self.gru = torch.nn.GRU(
                oc,
                oc // 2,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.conv_block1 = ConvBlock(
                oc, act_type=act_type, use_weight_norm=use_weight_norm
            )
            self.conv_block2 = ConvBlock(
                oc, act_type=act_type, use_weight_norm=use_weight_norm
            )
        else:
            raise ValueError("seq_model must be 'gru' or 'attention'")

    def forward(self, x, x_mel):
        # x: [B, n_channels, T_audio]
        # x_mel: [B, total_channels, T_mel]
        outputs = []
        lengths = []
        for idx, ds in enumerate(self.ds_modules):
            lengths.append(x.shape[-1])
            x, res, _ = ds(x)

            if self.st_convs[idx] is not None:
                res = self.st_convs[idx](res)
                outputs.append(res)
        outputs.append(x)

        # Combine them
        norm_factor = 1.0 / math.sqrt(len(outputs) + 1)
        out = x_mel
        for o in outputs:
            out = out + o
        out = out * norm_factor

        if self.seq_model == "gru":
            out, *_ = self.conv_block1(out)
            if self.with_gru_residual:
                res = out
            out, *_ = self.gru(out.transpose(-2, -1))  # B x T x C
            out = out.transpose(-2, -1)
            if self.with_gru_residual:
                out = (out + res) / math.sqrt(2)
            out, *_ = self.conv_block2(out)
        elif self.seq_model == "attention":
            out = self.att(out)
            
        return out, lengths[::-1]


class ConditionerDecoder(torch.nn.Module):
    def __init__(
        self,
        up_factors,
        input_channels,
        with_extra_conv_block=False,
        act_type="prelu",
        use_weight_norm=False,
        use_antialiasing=False,
    ):
        super().__init__()
        self.extra_conv_block = with_extra_conv_block

        n_channels = [
            input_channels * 2 ** (len(up_factors) - i - 1)
            for i in range(len(up_factors))
        ]
        self.input_conv_block = ConvBlock(
            n_channels[0] * 2, act_type=act_type, use_weight_norm=use_weight_norm
        )
        up_modules = [
            ConvBlock(
                c,
                r,
                "up",
                act_type=act_type,
                use_weight_norm=use_weight_norm,
                antialiasing=use_antialiasing,
            )
            for c, r in zip(n_channels, up_factors)
        ]
        if self.extra_conv_block:
            up_modules = [
                ConvBlock(
                    2 * n_channels[0],
                    act_type=act_type,
                    use_weight_norm=use_weight_norm,
                )
            ] + up_modules
        self.up_modules = torch.nn.ModuleList(up_modules)

    def forward(self, x, lengths):
        conditions = []
        x, *_ = self.input_conv_block(x)
        for up, length in zip(self.up_modules, lengths):
            x, _, cond = up(x, length=length)
            conditions.append(cond)
        return x, conditions


# ----------------------------------------------------------------------
#  NEW TEXT MODULES: FiLM & CrossAttention
# ----------------------------------------------------------------------
class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=4, temperature: float = 0.6): # 1.0 default (no temp)
        super().__init__()

        #  self.cross_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        ## ADD 8 MAY - DROPOUT ##
        # Add dropout to attention mechanism to prevent concentration
        self.cross_attn = torch.nn.MultiheadAttention(
            hidden_dim, 
            num_heads, 
            batch_first=True,
            dropout=0.1   # Add attention dropout
        )
        ## ADD 8 MAY - DROPOUT ##


        self.cross_attn = torch.nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
            dropout=0.0,        # ➌ keep dropout off
        )


        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm_ffn = torch.nn.LayerNorm(hidden_dim)
        
        
        # --- NEW 01 MAY
        # self.temperature = float(temperature)  # 1.0 → no change
        self.temperature = 1.0     # ➌ turn temperature scaling off # 09 MAY UPD
        if self.temperature <= 0.0:
            raise ValueError("attention_temperature must be > 0")
        print(f'Using cross-attn temperature: {self.temperature}')
        # --- NEW 01 MAY

        ## NEW 08 MAY ##
        # Initialize a local counter for dynamic temperature
        self.train_step_counter = 0
        self.temp_min_scale = 0.3   # Minimum temperature scale factor
        ## NEW 08 MAY ##


    def forward(self, x, cond, x_mask=None, cond_mask=None):
        text_metrics = {}
        
        # scale queries to adjust soft-max temperature ## NEW 01 MAY
        # q = x if self.temperature == 1.0 else x * (1.0 / self.temperature)
        
        # ## UPD 6 MAY v2#
        # scale = 1.0 / self.temperature
        # q     = x * scale if self.temperature != 1.0 else x          # queries
        # k     = cond * scale if self.temperature != 1.0 else cond    # keys
        # ## UPD 6 MAY v2#

        ## UPD 7 MAY ## 
        # Dynamic temperature that decreases as training progresses
        if self.training:
            # # Start hot (high temperature), gradually cool down
            # scale_factor = max(0.3, min(1.0, 1.0 - self.global_step / 50000))
            # effective_temp = self.temperature * scale_factor

            ## UPD 8 MAY ##
            # Increment the local counter - simpler than trying to access global_step
            self.train_step_counter += 1
            # Scale from 1.0 -> temp_min_scale over ~50k steps
            scale_factor = max(self.temp_min_scale, 
                             min(1.0, 1.0 - self.train_step_counter / 50000))
            ## UPD 8 MAY ##
            effective_temp = self.temperature * scale_factor
        else:
            effective_temp = self.temperature
        
        # scale = 1.0 / effective_temp
        # q = x * scale
        # k = cond * scale
        # ## UPD 7 MAY ## 


        q, k = x, cond # UPD 9 MAY - remove scaling


        attn_out, attn_weights = self.cross_attn(
            # x, cond, cond,
            # q, cond, cond,
            q, k, cond, # UPD 6 MAY v2
            key_padding_mask=cond_mask, # text
            need_weights=True, # ← keep the heads!
            average_attn_weights=False, # attn_weights: [B, H, Q, S]
        )
        
        ## UPD 6 MAY v2#
        # Head-diversity loss (skip for 1-head case)
        if self.training and attn_weights.size(1) > 1:
            B, H, Q, S = attn_weights.size()
            A   = attn_weights.view(B, H, -1)           # [B, H, Q*S]


            # sim = (A @ A.transpose(1,2)) / (Q * S)      # cosine-like similarity
            ## UPD 8 MAY ##
            # Normalize each head's attention for better diversity measurement
            A_norm = A / torch.norm(A, dim=2, keepdim=True).clamp_min(1e-8)
            sim = A_norm @ A_norm.transpose(1,2)        # true cosine similarity
            # UPD 8 MAY ##         

            self.div_loss = sim.triu(diagonal=1).mean() # upper-triangular avg
        else:
            self.div_loss = attn_weights.new_tensor(0.0)
        # ----------------------------------------------------------------
        ## UPD 6 MAY v2#        
        
        # ----------------------------------------------------------------------
        # 1) **per–head focus**  --------------  new metric & debug print
        # ----------------------------------------------------------------------
        #   max over tokens  →  mean over queries  →  mean over batch
        head_focus = attn_weights.max(dim=-1)[0].mean(dim=-1)      # [B, H]
        head_focus_mean = head_focus.mean(dim=0)                   # [H]
        print("[DEBUG] Attention focus per head:",
            [f"{v:.3f}" for v in head_focus_mean])
        text_metrics["attention_focus_per_head"] = head_focus_mean.tolist()
                
        ### 01 MAY ADD
 
        
        # ----------------------------------------------------------------------
        # 2) keep the *old* aggregate metrics — just average heads first
        # ----------------------------------------------------------------------
        attn_weights_avg = attn_weights.mean(dim=1)   # [B, Q, S]  (old shape)

        # --- ↓ replace every occurrence of `attn_weights` below with
        #       `attn_weights_avg` (the rest of the code stays identical) -------
        valid_q = (~x_mask).unsqueeze(-1) if x_mask is not None else None

        if x_mask is not None:
            attn_out = attn_out.masked_fill(x_mask.unsqueeze(-1), 0.0)
            max_attentions = (attn_weights_avg.masked_fill(~valid_q, -1)
                            .max(dim=-1)[0])
            attn_focus_metric = max_attentions[max_attentions >= 0].mean().item()
        else:
            attn_focus_metric = attn_weights_avg.max(dim=-1)[0].mean().item()

        text_metrics["attention_focus"] = attn_focus_metric
        
        ### 01 MAY ADD
        
        
        max_attentions = attn_weights.max(dim=-1)[0]
        print(f"[DEBUG] Attention focus: {attn_focus_metric:.4f}")
        print(f"[DEBUG] Attention weights stats: "
              f"min={attn_weights.min().item()}, max={attn_weights.max().item()}, mean={attn_weights.mean().item()}")

        # text_metrics["attention_focus"] = max_attentions.mean().item()
        text_metrics["attention_min"] = attn_weights.min().item()
        text_metrics["attention_max"] = attn_weights.max().item()
        text_metrics["attention_mean"] = attn_weights.mean().item()
        
      
        
        if cond_mask is not None:
            # broadcast to [B,H,Q,S] then reduce
            cond_mask_full = cond_mask.unsqueeze(1).unsqueeze(2)        # [B,1,1,S]
            valid_pos = ~cond_mask_full.expand_as(attn_weights)
            valid_att = attn_weights.masked_select(valid_pos)
            if valid_att.numel():
                text_metrics["valid_attention_focus"] = (
                    attn_weights.masked_fill(cond_mask_full, -1)
                    .max(dim=-1)[0].mean().item())
                text_metrics["valid_attention_min"]   = valid_att.min().item()
                text_metrics["valid_attention_max"]   = valid_att.max().item()
                text_metrics["valid_attention_mean"]  = valid_att.mean().item()
                text_metrics["padding_ratio"]         = (
                    cond_mask.float().sum() / cond_mask.numel()).item()

        
        # top-K indices (unchanged – but use attn_weights_avg)
        if attn_weights_avg.size(0):
            top_k = min(5, attn_weights_avg.size(-1))
            _, top_idx = torch.topk(attn_weights_avg[0].mean(dim=0), top_k)
            text_metrics["top_attended_positions"] = top_idx.tolist()
                

        x = x + attn_out
        x = self.layer_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.layer_norm_ffn(x)

        # return x, text_metrics
        return x, text_metrics, attn_weights_avg.detach()    #  ← extra tensor # NEW 03 MAY


class FiLM(torch.nn.Module):
    def __init__(self, condition_dim, feature_channels, init_scale=0.05): # 0.1 before # NEW 30 APR
        super().__init__()
        self.gamma_fc = torch.nn.Linear(condition_dim, feature_channels)
        self.beta_fc = torch.nn.Linear(condition_dim, feature_channels)
        self.scale = torch.nn.Parameter(torch.tensor(init_scale))

    def forward(self, x, cond):
        text_metrics = {}

        gamma = self.gamma_fc(cond).unsqueeze(1)  # (B,1,channels)
        beta = self.beta_fc(cond).unsqueeze(1)
        # print(f"[DEBUG] FiLM gamma stats: min={gamma.min().item():.4f}, max={gamma.max().item():.4f}")
        # print(f"[DEBUG] FiLM beta stats:  min={beta.min().item():.4f}, max={beta.max().item():.4f}")

        
        ### NEW 30 APR - clamp FILM gamma / beta
        
        # gamma = gamma.clamp(-10.0, 10.0)   
        # beta  = beta .clamp(-10.0, 10.0)
        
        gamma = 300 * torch.tanh(gamma / 300)
        beta  = 300 * torch.tanh(beta  / 300)

        
        ### NEW 30 APR - clamp FILM gamma / beta

        
        result = self.scale * (gamma * x + beta)
        # print(f"[DEBUG] FiLM input magnitude: {x.abs().mean().item():.4f}")
        # print(f"[DEBUG] FiLM output magnitude: {result.abs().mean().item():.4f}")

        text_metrics["film_gamma_min"] = gamma.min().item()
        text_metrics["film_gamma_max"] = gamma.max().item()
        text_metrics["film_beta_min"] = beta.min().item()
        text_metrics["film_beta_max"] = beta.max().item()
        text_metrics["film_input_magnitude"] = x.abs().mean().item()
        text_metrics["film_output_magnitude"] = result.abs().mean().item()

        return result, text_metrics



# ----------------------------------------------------------------------------
# New TextConditioner class
# ----------------------------------------------------------------------------
class TextConditioner(torch.nn.Module):
    # def __init__(self, text_encoder_config, film_global_dim, cross_attention_dim, total_channels):
    def __init__(self, text_encoder_config, film_global_dim, cross_attention_dim, total_channels,
        num_heads: int, attention_temperature: float):    
    
        super().__init__()
        # Instantiate user-provided text encoder (PL-BERT, etc.)
        self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
        print("[DEBUG] TextEncoder instantiated from config:", self.text_encoder)
        
        #### UPDATED in 512 VER ####
        self.cross_attention_dim = cross_attention_dim
        self.total_channels = total_channels
        #### UPDATED in 512 VER ####

        # FiLM
        self.film_global = FiLM(
            condition_dim=film_global_dim,
            feature_channels=total_channels
        )

        # Cross-attention
        self.mel_to_attn = torch.nn.Linear(total_channels, cross_attention_dim)
        # self.cross_attention = CrossAttentionBlock(cross_attention_dim, num_heads=4)
        # self.cross_attention = CrossAttentionBlock(cross_attention_dim, num_heads=cross_attention_dim // 64) # 512 → 8 heads
        self.cross_attention = CrossAttentionBlock(cross_attention_dim, num_heads=num_heads, temperature=attention_temperature)
        
        self.attn_to_mel = torch.nn.Linear(cross_attention_dim, total_channels)
        
        # ### ADD 06 MAY - BLANK TOKEN FOR TEXT
        # # --- (1) learnable BLANK token -----------------------------------
        # self.blank_emb = torch.nn.Parameter(torch.zeros(1, 1, cross_attention_dim))
        # torch.nn.init.trunc_normal_(self.blank_emb, std=0.02)
        # ### ADD 06 MAY - BLANK TOKEN FOR TEXT
        
        
        # ### ADD 04 MAY - POS EMBEDDING
        # # ---------- learnable ABSOLUTE positional emb. for the mel timeline ----
        # max_len = 1600                      # enough for 16 kHz / 128‑hop ≈ 10 s
        # self.pos_audio = torch.nn.Parameter(
        #     torch.zeros(1, max_len, cross_attention_dim)
        # )
        # torch.nn.init.trunc_normal_(self.pos_audio, std=0.02)
        # ### ADD 04 MAY - POS EMBEDDING


        ## ADD 09 MAY ##
        # --- learnable SIL token ------------------------------------------
        self.sil_emb = torch.nn.Parameter(torch.zeros(1, 1, cross_attention_dim))
        torch.nn.init.trunc_normal_(self.sil_emb, std=0.02)
        ## ADD 09 MAY ##

        
        ## ADD 08 MAY ##
        # Add learnable positional embeddings for text tokens
        self.max_text_positions = 256  # Maximum number of text tokens to support
        self.text_pos_embed = torch.nn.Parameter(
        torch.zeros(1, self.max_text_positions, cross_attention_dim)
            )
        torch.nn.init.trunc_normal_(self.text_pos_embed, std=0.02)
        ## ADD 08 MAY ##

        

        # init
        torch.nn.init.xavier_uniform_(self.mel_to_attn.weight)
        torch.nn.init.zeros_(self.mel_to_attn.bias)
        torch.nn.init.xavier_uniform_(self.attn_to_mel.weight)
        torch.nn.init.zeros_(self.attn_to_mel.bias)

        # impact factor
        self.text_impact_factor = torch.nn.Parameter(torch.tensor(0.3))
        print("[DEBUG] FiLM + cross-attention for text conditioning ready.")
        

        print("[DEBUG] FiLM + cross-attention for text conditioning ready.")
        
        
    # ### ADD 06 MAY - ROPE POS EMBEDDING
    # # ---- (2) Rotary Positional Embedding  --------------------------
    # @staticmethod
    # def _rope(t, base=10000):
    #         # t : [B,T,C] (C even)
    #         half = t[..., ::2], t[..., 1::2]
    #         freqs = torch.arange(0, half[0].shape[-1], device=t.device)
    #         freqs = 1.0 / (base ** (freqs / half[0].shape[-1]))
    #         angles = torch.einsum('t,d->td', torch.arange(t.shape[1], device=t.device),
    #                               freqs).unsqueeze(0)
    #         sin, cos = angles.sin(), angles.cos()
    #         return torch.cat([half[0]*cos - half[1]*sin,
    #                           half[0]*sin + half[1]*cos], -1)

    # ### ADD 06 MAY - ROPE POS EMBEDDING 

    ### UPD 8 MAY - IMRPOVE ROPE
    # ---- (2) Rotary Positional Embedding  --------------------------
    @staticmethod
    def _rope(t, base=10000):
        # Enhanced RoPE with better frequency allocation
        # t : [B,T,C] (C even)
        d = t.shape[-1]
        half = t[..., :d//2], t[..., d//2:]
        
        # Use log-spaced frequencies for better representation
        dim_t = torch.arange(0, d//2, device=t.device)
        inv_freq = 1.0 / (base ** (dim_t / (d//2)))
        
        # Generate position-dependent rotation
        seq_len = t.shape[1]
        pos = torch.arange(seq_len, device=t.device).float().unsqueeze(1)  # [T, 1]
        
        # Compute angles
        angles = pos * inv_freq.unsqueeze(0)  # [T, d//2]
        sin, cos = angles.sin().unsqueeze(0), angles.cos().unsqueeze(0)  # [1, T, d//2]
        
        # Apply rotation
        return torch.cat([
            half[0] * cos - half[1] * sin,
            half[0] * sin + half[1] * cos
        ], dim=-1)
    ### UPD 8 MAY - IMRPOVE ROPE
        

    def forward(self, x_mel, text, q_pad_mask=None): ### 01 MAY: ADD MASK
        """
        Applies text conditioning (FiLM + cross-attn) to x_mel given text.
        Returns the conditioned x_mel and a dictionary of metrics.
        """
        # print("[DEBUG] Text conditioning is active in condition_plbert.")
        text_metrics = {}
        x_mel_orig = x_mel.clone()
        
        # 1) Encode text => (global_emb, seq_emb)
        # global_emb, seq_emb = self.text_encoder(text)
        
        global_emb, seq_emb, text_key_mask = self.text_encoder(text) # [B,S,C]  mask: [B,S]
        # T_text = seq_emb.size(1)
        print(f"[DEBUG] Text key mask tokens inside TE: {text_key_mask.sum(dim=1).tolist()}")
        
        
        # 2) FiLM on x_mel
        x_mel_t = x_mel.transpose(1, 2)  # => [B, T_mel, 512]
        x_mel_t, film_info = self.film_global(x_mel_t, global_emb)
        text_metrics.update(film_info)

        # 3) Cross-attn
        # x_mel_attn = self.mel_to_attn(x_mel_t)   # => [B, T_mel, cross_attention_dim]
        
        #### UPDATED in 512 VER ####
        if self.total_channels != self.cross_attention_dim:
            x_mel_attn = self.mel_to_attn(x_mel_t)   # => [B, T_mel, cross_attention_dim]
        else:
            x_mel_attn = x_mel_t
        #### UPDATED in 512 VER ####
        
        
        # #### ADD 04 MAY - POS EMBEDDING ####
        # # ── add positional encoding to the *queries* (mel timeline) ───────────
        # T = x_mel_attn.size(1)
        # pos_emb = self.pos_audio[:, :T, :]           # trim if sequence shorter
        # x_mel_attn = x_mel_attn + pos_emb
        # #### ADD 04 MAY - POS EMBEDDING ####
        
        ### ADD 06 MAY - ROPE POS EMBEDDING


        x_mel_attn = self._rope(x_mel_attn)
        ### ADD 06 MAY - ROPE POS EMBEDDING
        
        
        
        
        # x_mel_attn, attn_info = self.cross_attention(x_mel_attn, seq_emb)
        # x_mel_attn, attn_info = self.cross_attention(x_mel_attn, seq_emb, cond_mask = text_key_mask)
        
        # x_mel_attn, attn_info = self.cross_attention(x_mel_attn, seq_emb, x_mask=q_pad_mask, cond_mask = text_key_mask) ### 01 MAY: add x_mask
        
        
        # x_mel_attn, attn_info, attn_map = self.cross_attention(x_mel_attn, seq_emb, x_mask=q_pad_mask, cond_mask=text_key_mask) ### 03 MAY ADD
        # # self.last_attn_map = attn_map           #  [B, Q, S] – handy for logging                                                ### 03 MAY ADD
        
        
        # ### ADD 06 MAY - UPD THIS WITH BLANK TOKEN
        # # ---- (3) prepend / append BLANK token --------------------------
        # B = seq_emb.size(0)
        # blank = self.blank_emb.expand(B, -1, -1)
        # seq_emb = torch.cat([blank, seq_emb, blank], 1)
        # text_key_mask = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=seq_emb.device),
        #                            text_key_mask,
        #                            torch.zeros(B,1,dtype=torch.bool,device=seq_emb.device)],1)
        # ### ADD 06 MAY - UPD THIS WITH BLANK TOKEN



        ## ADD 09 MAY ##
        # ---- prepend single SIL token (un-masked) --------------------------
        B = seq_emb.size(0)
        sil = self.sil_emb.expand(B, -1, -1)          # [B,1,C]
        seq_emb      = torch.cat([sil, seq_emb], 1)   # new length = S+1
        text_key_mask = torch.cat(
            [torch.zeros(B,1, dtype=torch.bool, device=seq_emb.device),  # False (= keep) for SIL
             text_key_mask],
            1)
        ## ADD 09 MAY ##

        

        ## ADD 08 MAY - POS EMB FOR TEXT
        # Add positional embeddings to text tokens
        seq_len = seq_emb.size(1)  # now includes SIL
        if seq_len <= self.max_text_positions:
            # Add positional embeddings only to real tokens (not padding)
            pos_emb = self.text_pos_embed[:, :seq_len, :]
            # Apply only to valid tokens (including blanks which are marked valid)
            seq_emb = seq_emb + pos_emb * (~text_key_mask).unsqueeze(-1).float()
        else:
           # Handle case where sequence exceeds max positions
            pos_emb = self.text_pos_embed[:, :self.max_text_positions, :]
            seq_emb[:, :self.max_text_positions, :] = (
                seq_emb[:, :self.max_text_positions, :] + 
                pos_emb * (~text_key_mask[:, :self.max_text_positions]).unsqueeze(-1).float()
            )
        ## ADD 08 MAY - POS EMB FOR TEXT


        # ---- (4) zero padded queries to avoid bleed --------------------
        if q_pad_mask is not None:
            x_mel_attn = x_mel_attn.masked_fill(q_pad_mask.unsqueeze(-1), 0.)

        x_mel_attn, attn_info, attn_map = self.cross_attention(
            x_mel_attn, seq_emb, x_mask=q_pad_mask, cond_mask=text_key_mask)
        
        ### ADD 06 MAY - UPD THIS WITH BLANK TOKEN
        
        
        
        
        #### ADD 05 MAY ####
        # ----- cache everything needed for GA loss ------------------
        self.last_attn_map = attn_map.detach()     # [B,Q,S]
        # “valid” == non‑padded
        # self.last_q_mask   = (q_pad_mask is None) \
        #                        or (~q_pad_mask)    # [B,Q]  bool
        
        
        ## 06 MAY SMALL FIX
        if q_pad_mask is None:
            self.last_q_mask = torch.ones_like(x_mel_attn[...,0], dtype=torch.bool)  # [B,Q]
        else:
            self.last_q_mask = ~q_pad_mask
        
        # ## 06 MAY v2 FIX ##    
        # # ── mark real text tokens (blanks & padding = False) ───────────────
        # valid_token = ~text_key_mask.clone()      # start with regular inversion
        # valid_token[:, 0]  = False                # left blank column
        # valid_token[:, -1] = False                # right blank column
        # self.last_s_mask = valid_token            # replaces old assignment
        # ## 06 MAY v2 FIX ##   

        # # real tokens = not padding (we no longer have blanks)
        # self.last_s_mask = ~text_key_mask


        # real tokens = not padding and not the SIL column (idx 0)
        valid = ~text_key_mask.clone()
        valid[:, 0] = False                      # exclude SIL from GA bookkeeping
        self.last_s_mask = valid

        ### 09 MAY UPD ##
        #  -- GA/Cover losses disabled for now ------------------------
        # self/last_attn_map = None
        ### 09 MAY UPD ##
                    
            
        ## 06 MAY SMALL FIX
        
        # self.last_s_mask   = ~text_key_mask        # [B,S]  bool
        #### ADD 05 MAY ####
        
        
        text_metrics.update(attn_info)
        
        # x_mel_t = self.attn_to_mel(x_mel_attn)   # => [B, T_mel, 512]
        
        #### UPDATED in 512 VER ####
        if self.total_channels != self.cross_attention_dim:
            x_mel_t = self.attn_to_mel(x_mel_attn)   # => [B, T_mel, 512]
        else:
            x_mel_t = x_mel_attn
        #### UPDATED in 512 VER ####

        # 4) L2 norm
        x_mel_norm = (x_mel_t.transpose(1,2)**2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).sqrt()
        x_mel_conditioned = x_mel_t.transpose(1,2) / x_mel_norm.clamp(min=1e-5)

        # 5) Blend with text_impact_factor
        blend_factor = torch.sigmoid(self.text_impact_factor)
        print(f"[DEBUG] blend_factor={blend_factor.item():.4f}")
        x_mel = (1.0 - blend_factor) * x_mel_orig + blend_factor * x_mel_conditioned

        # match magnitude
        new_norm = x_mel.norm(dim=1, keepdim=True)
        old_norm = x_mel_orig.norm(dim=1, keepdim=True)
        x_mel = x_mel * (old_norm / new_norm.clamp(min=1e-8))

        text_metrics["blend_factor"] = blend_factor.item()
        text_metrics["mel_features_before"] = x_mel_orig.abs().mean().item()
        text_metrics["mel_features_after"] = x_mel.abs().mean().item()
        text_metrics["feature_difference"] = (x_mel - x_mel_orig).abs().mean().item()

        return x_mel, text_metrics


# ----------------------------------------------------------------------
#  FINAL MERGED ConditionerNetwork with optional text conditioning
# ----------------------------------------------------------------------
class ConditionerNetwork(torch.nn.Module):
    """
    This merges the old condition.py structure with new text logic.
    If text_encoder_config is None or text is empty => old path (unchanged).
    If text is present => new FiLM + cross-attention + debug prints.
    """

    def __init__(
        self,
        fb_kernel_size=3,
        rate_factors=[2, 4, 4, 5],
        n_channels=32,
        n_mels=80,
        n_mel_oversample=4,
        encoder_gru_residual=False,
        extra_conv_block=False,
        encoder_act_type="prelu",
        decoder_act_type="prelu",
        precoding=None,
        input_channels=1,
        # optional, if specified, an extra conv. layer is used as adapter
        # for the output signal estimat y_est
        output_channels=None,
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        # New text config
        text_encoder_config=None,  # If None => skip text logic
        film_global_dim=512,       # dimension for global text embedding # 512 is better here       #### UPDATED in 512 VER ####
        cross_attention_dim=512,    # dimension for cross-attn # 512 is better here                  #### UPDATED in 512 VER ####
        text_lr_scale=1.0,           # NEW 30 APR
        cross_attention_num_heads=None,          # ← NEW
        attention_temperature=0.6               # ← NEW
    ):
        super().__init__()
        self.input_conv = cond_weight_norm(
            torch.nn.Conv1d(
                input_channels, n_channels, kernel_size=fb_kernel_size, padding="same"
            ),
            use=use_weight_norm,
        )
        
        self.text_lr_scale = text_lr_scale #### NEW 30 APR

        if output_channels is not None:
            self.output_conv = cond_weight_norm(
                torch.nn.Conv1d(
                    n_channels,
                    output_channels,
                    kernel_size=fb_kernel_size,
                    padding="same",
                ),
                use=use_weight_norm,
            )
        else:
            self.output_conv = None

        total_ds = math.prod(rate_factors)
        total_channels = 2 ** len(rate_factors) * n_channels  # e.g. 512
        self.total_channels = total_channels
        self.input_mel = MelAdapter(
            n_mels,
            total_channels,
            total_ds * input_channels,
            n_mel_oversample,
            use_weight_norm=use_weight_norm,
        )

        self.encoder = ConditionerEncoder(
            rate_factors,
            n_channels,
            with_gru_residual=encoder_gru_residual,
            with_extra_conv_block=extra_conv_block,
            act_type=encoder_act_type,
            use_weight_norm=use_weight_norm,
            seq_model=seq_model,
            # use_antialiasing=use_antialiasing,
            use_antialiasing=False, # FIX 10 MAY TO MATCH ORIGINAL VER
        )
        self.decoder = ConditionerDecoder(
            rate_factors[::-1],
            n_channels,
            with_extra_conv_block=extra_conv_block,
            act_type=decoder_act_type,
            use_weight_norm=use_weight_norm,
            use_antialiasing=use_antialiasing,
        )

        self.precoding = instantiate(precoding, _recursive_=True) if precoding else None

        # ----------------------------------------------------
        # Now handle text logic by creating a TextConditioner or not
        # ----------------------------------------------------
        if text_encoder_config is not None:
            self.text_conditioner = TextConditioner(
                text_encoder_config,
                film_global_dim,
                cross_attention_dim,
                total_channels,
                num_heads=cross_attention_num_heads
                       or max(1, cross_attention_dim // 64),
                attention_temperature=attention_temperature,
            )
        else:
            self.text_conditioner = None
            print("[DEBUG] No text_encoder_config => skipping text features.")
            
        
        # # ——— gradient‐hook bookkeeping ————————————————————————————
       
        # Make sure the log file exists and is writable
        self.grad_log_path = os.path.join(os.getcwd(), "grad_stats.log")
        
        # Create an empty file if it doesn't exist
        with open(self.grad_log_path, "a") as f:
            f.write(f"# Gradient stats log created at {datetime.datetime.now()}\n")
        
        print(f"[INFO] Gradient stats will be logged to: {self.grad_log_path}")
        
        # Simplify the grad stats tracking to just a dictionary
        self.grad_stats = defaultdict(list)
        
        # Simplified hook without trying to track multiple statistics
        def make_module_hook(mod_name):
            def hook(module, grad_input, grad_output):
                if not grad_output or grad_output[0] is None:
                    return
                    
                g = grad_output[0].detach()
                l2_norm = g.norm().item()
                max_val = g.abs().max().item()
                
                # Store in memory
                self.grad_stats[mod_name].append((l2_norm, max_val))
                
                # Make sure the grad_stats.log exists in current directory
                with open(self.grad_log_path, "a") as f:
                    f.write(f"{mod_name}\t{l2_norm:.6e}\t{max_val:.6e}\n")
            return hook
            
        # Register hooks on key modules
        for name, module in self.named_modules():
            if isinstance(module, (ConvBlock, PReLU_Conv, torch.nn.Linear)):
                module.register_full_backward_hook(make_module_hook(name))

    # def forward(self, x, x_wav=None, train=False, text=None):
    def forward(self, x, x_wav=None, train=False, text=None,
                mask: torch.Tensor | None = None): # NEW 01 MAY   
        
        """
        If text is None or empty => old path. 
        If text present => do FiLM/cross-attn. 
        Return the same outputs as old code, plus text_metrics if text is used.
        """
        n_samples = x.shape[-1]
        if x_wav is None:
            # this is used in case some type of transform is appled to
            # x before input.
            # This way, we can pass the original waveform
            x_wav = x

        # Compute mel features
        x_mel = self.input_mel(x_wav)  # => [B, total_channels, T_mel]

        text_metrics = {}
        # Decide if we do text logic
        use_text = (
            # self.text_encoder is not None
            self.text_conditioner is not None
            and text is not None
            and any(t.strip() for t in text)
        )
        
        
        
            
        ##### 01 May add    
        if mask is not None:
            L_in, L_mel = mask.shape[-1], x_mel.shape[-1]          # wav vs Mel length
            f_mel = math.ceil(L_in / L_mel)                        # robust ratio
            mel_pad_mask = (torch.nn.functional.avg_pool1d(mask.unsqueeze(1).float(),
                                        kernel_size=f_mel,
                                        stride=f_mel,
                                        ceil_mode=True)           # keep tail
                            .squeeze(1) < 0.5)                   # True → padded
            mel_pad_mask = mel_pad_mask[..., :L_mel]               # trim excess
        else:
            mel_pad_mask = None
        ##### 01 May add    
        
        
        ##### TextConditioner - right after MelAdapter ##### 01 MAY ADD (second)
        if use_text:
            # Call the new TextConditioner class
            x_mel, text_metrics = self.text_conditioner(x_mel, text, mel_pad_mask) ### 01 MAY ADD
            print("[DEBUG] Use text iS TRUE")
        else:
            # old path => do nothing special, x_mel remains as is
            print("[DEBUG] No text => old conditioning path in condition_plbert.")
        ##### TextConditioner - right after MelAdapter #####

            
        if self.precoding:
            x = self.precoding(x) # do this after mel-spec comp

        # old code: main forward
        x = self.input_conv(x)
        h, lengths = self.encoder(x, x_mel) # latent representation
        
        #### 01 May add

        if mask is not None:
            # --- robust down-sampling -----------------------------------------
            L_in, L_lat = mask.shape[-1], h.shape[-1]
            factor = math.ceil(L_in / L_lat)                 # ← use CEIL not //
            q_pad_mask = (torch.nn.functional.avg_pool1d(
                            mask.unsqueeze(1).float(),
                            kernel_size=factor,
                            stride=factor,
                            ceil_mode=True)                # keep last partial window
                        .squeeze(1) < 0.5)                 # bool: True = pad
            q_pad_mask = q_pad_mask[..., :L_lat]             # trim any extra frame
        else:
            q_pad_mask = None    
        
        #### 01 May add
        
        
        
        ##### TextConditioner - right at end of ConditionerEncoder (right after it) #####
        if use_text:
            # Call the new TextConditioner class
            # h, text_metrics = self.text_conditioner(h, text)
            h, text_metrics = self.text_conditioner(h, text, q_pad_mask)    ### 01 MAY ADD
        else:
            # old path => do nothing special, x_mel remains as is
            print("[DEBUG] No text => old conditioning path in condition_plbert.")
        ##### TextConditioner - right at end of ConditionerEncoder (right after it) #####
        
        
        y_hat, conditions = self.decoder(h, lengths)

        if self.output_conv is not None:
            y_hat = self.output_conv(y_hat)

        if self.precoding:
            y_hat = self.precoding.inv(y_hat)

        # adjust length and dimensions
        y_hat = torch.nn.functional.pad(y_hat, (0, n_samples - y_hat.shape[-1]))

        # Return old structure plus text_metrics if used
        if train:
            if use_text:
                return conditions, y_hat, h, text_metrics
            else:
                return conditions, y_hat, h
        else:
            if use_text:
                return conditions, text_metrics
            else:
                return conditions
            
    def dump_grad_stats(self):
        """Write a summary of gradient statistics to the log file."""
        try:
            print(f"[DEBUG] Dumping gradient stats to {self.grad_log_path}")
            
            with open(self.grad_log_path, "a") as f:
                f.write(f"\n--- Batch summary at {datetime.datetime.now()} ---\n")
                
                for mod_name, stats_list in self.grad_stats.items():
                    if not stats_list:
                        continue
                        
                    # Get the last stats
                    l2_norm, max_val = stats_list[-1]
                    f.write(f"{mod_name}\tL2={l2_norm:.6e}\tMax={max_val:.6e}\n")
                    
            # Clear the stats for the next batch
            self.grad_stats.clear()
            print(f"[DEBUG] Gradient stats dumped successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to dump gradient stats: {e}")

