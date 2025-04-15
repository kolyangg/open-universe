# condition_plbert.py
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
import torch.nn.functional as F

from .blocks import (
    BinomialAntiAlias,
    ConvBlock,
    PReLU_Conv,
    cond_weight_norm,
)


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
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm_ffn = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, cond, x_mask=None, cond_mask=None):
        text_metrics = {}

        attn_out, attn_weights = self.cross_attn(
            x, cond, cond,
            attn_mask = x_mask, # audio-side
            key_padding_mask = cond_mask, # text
            need_weights=True
        )
        max_attentions = attn_weights.max(dim=-1)[0]
        print(f"[DEBUG] Attention focus: {max_attentions.mean().item():.4f}")
        print(f"[DEBUG] Attention weights stats: "
              f"min={attn_weights.min().item()}, max={attn_weights.max().item()}, mean={attn_weights.mean().item()}")

        text_metrics["attention_focus"] = max_attentions.mean().item()
        text_metrics["attention_min"] = attn_weights.min().item()
        text_metrics["attention_max"] = attn_weights.max().item()
        text_metrics["attention_mean"] = attn_weights.mean().item()

        if attn_weights.shape[0] > 0:
            # topK for first sample
            attn_sample = attn_weights[0]
            top_k = min(5, attn_sample.shape[-1])
            _, top_indices = torch.topk(attn_sample.mean(dim=0), top_k)
            print(f"[DEBUG] Top {top_k} attended positions: {top_indices.tolist()}")
            text_metrics["top_attended_positions"] = top_indices.tolist()

        x = x + attn_out
        x = self.layer_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.layer_norm_ffn(x)

        return x, text_metrics


class FiLM(torch.nn.Module):
    def __init__(self, condition_dim, feature_channels, init_scale=0.01): # 0.1 before
        super().__init__()
        self.gamma_fc = torch.nn.Linear(condition_dim, feature_channels)
        self.beta_fc = torch.nn.Linear(condition_dim, feature_channels)
        
        # NEW FOR LOWER GRAD: Add these lines for explicit initialization
        torch.nn.init.normal_(self.gamma_fc.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.gamma_fc.bias)
        torch.nn.init.normal_(self.beta_fc.weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.beta_fc.bias)
        
        self.scale = torch.nn.Parameter(torch.tensor(init_scale))

    def forward(self, x, cond):
        text_metrics = {}

        gamma = self.gamma_fc(cond).unsqueeze(1)  # (B,1,channels)
        beta = self.beta_fc(cond).unsqueeze(1)
        print(f"[DEBUG] FiLM gamma stats: min={gamma.min().item():.4f}, max={gamma.max().item():.4f}")
        print(f"[DEBUG] FiLM beta stats:  min={beta.min().item():.4f}, max={beta.max().item():.4f}")

        result = self.scale * (gamma * x + beta)
        print(f"[DEBUG] FiLM input magnitude: {x.abs().mean().item():.4f}")
        print(f"[DEBUG] FiLM output magnitude: {result.abs().mean().item():.4f}")

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
    def __init__(self, text_encoder_config, film_global_dim, cross_attention_dim, total_channels):
        super().__init__()
        # Instantiate user-provided text encoder (PL-BERT, etc.)
        self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
        print("[DEBUG] TextEncoder instantiated from config:", self.text_encoder)

        # FiLM
        self.film_global = FiLM(
            condition_dim=film_global_dim,
            feature_channels=total_channels
        )

        # Cross-attention
        self.mel_to_attn = torch.nn.Linear(total_channels, cross_attention_dim)
        self.cross_attention = CrossAttentionBlock(cross_attention_dim, num_heads=4)
        
        # NEW FOR LOWER GRAD
        # Add explicit initialization for cross-attention
        for name, param in self.cross_attention.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        
        self.attn_to_mel = torch.nn.Linear(cross_attention_dim, total_channels)

        # init
        torch.nn.init.xavier_uniform_(self.mel_to_attn.weight)
        torch.nn.init.zeros_(self.mel_to_attn.bias)
        torch.nn.init.xavier_uniform_(self.attn_to_mel.weight)
        torch.nn.init.zeros_(self.attn_to_mel.bias)

        # impact factor
        self.text_impact_factor = torch.nn.Parameter(torch.tensor(0.3))
        print("[DEBUG] FiLM + cross-attention for text conditioning ready.")

    def forward(self, x_mel, text, mask = None):
        """
        Applies text conditioning (FiLM + cross-attn) to x_mel given text.
        Returns the conditioned x_mel and a dictionary of metrics.
        """
        print("[DEBUG] Text conditioning is active in condition_plbert.")
        print(f"[DEBUG] mask: {mask}")
        text_metrics = {}
        x_mel_orig = x_mel.clone()
        
        if mask is not None:
            # Get shapes
            B, T_mel, _ = x_mel.shape
        
            # Get text length safely
            if hasattr(x_mel, 'shape'):
                _, T_text, _ = x_mel.shape
            else:
                # If cond is a list or other non-tensor
                T_text = len(x_mel[0]) if isinstance(x_mel, list) and len(x_mel) > 0 else 1
                
            # Calculate the approximate downsampling ratio
            # From audio samples (64000) to mel frames (401)
            audio_len = mask.shape[1]
            downsample_ratio = audio_len / T_mel
            
            # Reshape for pooling: [B, 1, audio_len]
            x_mask_reshaped = mask.float().unsqueeze(1)
            
            # Use avg_pool1d to downsample the mask
            # The kernel size should be approximately the downsample ratio
            kernel_size = int(downsample_ratio)
            stride = kernel_size
            
            # Ensure kernel size is at least 1
            kernel_size = max(1, kernel_size)
            
            # Perform the downsampling
            downsampled_mask = F.avg_pool1d(
                x_mask_reshaped, 
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )
            
            # If the resulting mask isn't exactly the right length, 
            # use interpolate to fix it
            if downsampled_mask.shape[2] != T_mel:
                downsampled_mask = F.interpolate(
                    downsampled_mask,
                    size=T_mel,
                    mode='linear',
                    align_corners=False
                )
            
            # Convert back to binary mask (1 = keep, 0 = mask)
            downsampled_mask = (downsampled_mask > 0.5).float()
            
            # For PyTorch attention, we need key_padding_mask where True = mask out
            # Converting [B, 1, T_mel] -> [B, T_mel] and inverting (1->0, 0->1)
            audio_key_padding_mask = ~(downsampled_mask.squeeze(1).bool())
            
            # Debug info
            print(f"[DEBUG] Downsampled: {audio_key_padding_mask.shape}")
            
            
        
        # 1) Encode text => (global_emb, seq_emb)
        # global_emb, seq_emb = self.text_encoder(text)
        global_emb, seq_emb, text_key_mask = self.text_encoder(text)
        T_text = seq_emb.size(1)
        
        # 2) FiLM on x_mel
        x_mel_t = x_mel.transpose(1, 2)  # => [B, T_mel, 512]
        x_mel_t, film_info = self.film_global(x_mel_t, global_emb)
        text_metrics.update(film_info)
        
        attn_mask = None
        if mask is not None:
            B, audio_len = mask.shape
            T_mel = x_mel_t.size(1)

            mask = mask.unsqueeze(1).float()  # => [B,1,audio_len]
            mask_ds = torch.nn.functional.interpolate(
                mask, size=T_mel, mode='nearest'
            )  # => [B,1,T_mel]
            # binarize
            mask_ds = (mask_ds > 0.5)
            
            # invert => True => block
            mask_ds = ~mask_ds  # shape [B,1,T_mel]
            
            # expand to [B,T_mel,T_text], etc. if you want (T_mel,T_text)
            T_text = seq_emb.size(1)
            mask_ds = mask_ds.transpose(-2,-1)   # => [B,T_mel,1]
            attn_mask = mask_ds.expand(-1, T_mel, T_text)  # => [B,T_mel,T_text]
            
            n_heads = 4
            attn_mask = attn_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
            # shape => [B, n_heads, T_q, T_k]

            B, _, T_q, T_k = attn_mask.shape
            attn_mask = attn_mask.reshape(B * n_heads, T_q, T_k)

        # 3) Cross-attn
        x_mel_attn = self.mel_to_attn(x_mel_t)   # => [B, T_mel, cross_attention_dim]
        
        # print(f"[DEBUG] Audio features shape: {x_mel_attn.shape}")
        # print(f"[DEBUG] Text features shape: {seq_emb.shape}")
        # print(f"[DEBUG] Audio mask shape: {mask.shape if mask is not None else None}")
        # print(f"[DEBUG] Text mask shape: {text_key_mask.shape if text_key_mask is not None else None}")
         
        
        # x_mel_attn, attn_info = self.cross_attention(x_mel_attn, seq_emb, x_mask = audio_key_padding_mask, cond_mask = text_key_mask)
        x_mel_attn, attn_info = self.cross_attention(x_mel_attn, seq_emb, x_mask=attn_mask)
        text_metrics.update(attn_info)
        x_mel_t = self.attn_to_mel(x_mel_attn)   # => [B, T_mel, 512]
        
        
        # if mask:          
        #     # Now call cross-attention properly:
        #     x_mel_attn, attn_info = self.cross_attention(
        #         x_mel_attn, seq_emb, seq_emb,
        #         cond_mask=text_key_mask,  # For text (keys/values)
        #         x_mask=None, # attn_mask,  # Not using attn_mask
        #         need_weights=True
        #     )
            
        #     # Store the mask for logging
        #     # text_metrics["audio_valid_ratio"] = mask_ds.float().mean().item()
        # else:
        #     # No mask provided
        #     x_mel_attn, attn_info = self.cross_attention(
        #         x_mel_attn, seq_emb, seq_emb,
        #         key_padding_mask=text_key_mask,
        #         need_weights=True
        #     )
                        

        # # 3b) Cross-attn with your attn_mask for the query side
        # x_mel_attn, attn_info = self.cross_attention(
        #     x_mel_attn, seq_emb, x_mask=attn_mask
        # )
        text_metrics.update(attn_info)

        x_mel_t = self.attn_to_mel(x_mel_attn)  # => [B, T_mel, 512]
        
        #### NEW SECTION TO FIX MASK ####

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
        film_global_dim=256,       # dimension for global text embedding # 512 is better here
        cross_attention_dim=256    # dimension for cross-attn # 512 is better here
    ):
        super().__init__()
        self.input_conv = cond_weight_norm(
            torch.nn.Conv1d(
                input_channels, n_channels, kernel_size=fb_kernel_size, padding="same"
            ),
            use=use_weight_norm,
        )

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
            use_antialiasing=use_antialiasing,
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
                total_channels
            )
        else:
            self.text_conditioner = None
            print("[DEBUG] No text_encoder_config => skipping text features.")

    def forward(self, x, x_wav=None, train=False, text=None, mask =None):
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

            
        if self.precoding:
            x = self.precoding(x) # do this after mel-spec comp

        # old code: main forward
        x = self.input_conv(x)
        h, lengths = self.encoder(x, x_mel) # latent representation
        
        
        ##### TextConditioner - right at end of ConditionerEncoder (right after it) #####
        if use_text:
            # Call the new TextConditioner class
            h, text_metrics = self.text_conditioner(h, text, mask)
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

