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
Conditioner network for the UNIVERSE model.

Author: Robin Scheibler (@fakufaku)
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
        x = self.mel_spec(x)
        x = x.squeeze(1)

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
            raise ValueError("Values for 'seq_model' can be gru|attention")

    def forward(self, x, x_mel):
        outputs = []
        lengths = []
        for idx, ds in enumerate(self.ds_modules):
            lengths.append(x.shape[-1])

            x, res, _ = ds(x)

            if self.st_convs[idx] is not None:
                res = self.st_convs[idx](res)
                outputs.append(res)
        outputs.append(x)

        norm_factor = 1.0 / math.sqrt(len(outputs) + 1)
        out = x_mel
        for o in outputs:
            out = out + o
        out = out * norm_factor

        if self.seq_model == "gru":
            out, *_ = self.conv_block1(out)
            if self.with_gru_residual:
                res = out
            out, *_ = self.gru(out.transpose(-2, -1))
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

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        """
        Args:
            hidden_dim: dimension of the queries/keys/values for cross-attention.
            num_heads: number of attention heads.
        """
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Add input layer norms for better gradient flow
        self.pre_attn_norm = torch.nn.LayerNorm(hidden_dim)
        self.pre_cond_norm = torch.nn.LayerNorm(hidden_dim)
        
        # Post-attention layer norm
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
        # Improved FFN with dropout
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),  # GELU often works better than ReLU
            torch.nn.Dropout(0.1),  # Add dropout for regularization
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm_ffn = torch.nn.LayerNorm(hidden_dim)
        
        # Temperature parameter to control attention sharpness
        self.attention_temperature = torch.nn.Parameter(torch.tensor(0.1))
        
        # Residual connection scale factor
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.8))

    def forward(self, x, cond, x_mask=None, cond_mask=None):
        text_metrics = {} # for wandb logging
        
        # Apply layer normalization before attention
        x_norm = self.pre_attn_norm(x)
        cond_norm = self.pre_cond_norm(cond)
        
        # Run cross-attention with normalized inputs
        attn_out, attn_weights = self.cross_attn(
            x_norm, cond_norm, cond_norm,
            key_padding_mask=cond_mask,
            need_weights=True
        )
        
        
        # Add noise to prevent attention collapse
        if self.training:
            noise_scale = 0.1 * torch.exp(-0.1 * torch.tensor(getattr(self, 'global_step', 0.0)))
            attn_weights = attn_weights + torch.rand_like(attn_weights) * noise_scale
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        
        # Apply temperature to sharpen attention
        # Track original weights for debugging
        orig_attn_max = attn_weights.max(dim=-1)[0].mean().item()
        
        # Scale attention weights by temperature (lower temp = sharper focus)
        # But only after a few steps to allow initial exploration
        temp = self.attention_temperature.clamp(min=0.01, max=1.0)
        
        # Record metrics before any modifications
        attn_weights_mean_per_head = attn_weights.mean(dim=1)  # Average across sequence
        max_attentions = attn_weights.max(dim=-1)[0]  # Max attention per position
        print(f"[DEBUG] Attention focus: {max_attentions.mean().item():.4f} (higher = more focused)")
        print(f"[DEBUG] Attention temperature: {temp.item():.4f} (lower = sharper focus)")
        print(f"[DEBUG] Attention weights stats: min={attn_weights.min().item():.4f}, " 
            f"max={attn_weights.max().item():.4f}, mean={attn_weights.mean().item():.4f}")
        
        text_metrics["attention_focus"] = max_attentions.mean().item()
        text_metrics["attention_temp"] = temp.item()
        text_metrics["attention_min"] = attn_weights.min().item()
        text_metrics["attention_max"] = attn_weights.max().item()
        text_metrics["attention_mean"] = attn_weights.mean().item()
        
        # After computing attention weights
        batch_size = attn_weights.shape[0]
        if batch_size > 0:  # Check to avoid empty batch issues
            # Get most attended positions for the first sample in batch
            attn_sample = attn_weights[0]  # First sample in batch
            top_k = min(5, attn_sample.shape[-1])
            _, top_indices = torch.topk(attn_sample.mean(dim=0), top_k)
            print(f"[DEBUG] Top {top_k} attended positions: {top_indices.tolist()}")
            text_metrics["top_attended_positions"] = top_indices.tolist()
                
        # Controlled residual connection with scaling
        res_scale = self.residual_scale.clamp(min=0.5, max=1.0)
        x = x + res_scale * attn_out
        x = self.layer_norm(x)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = x + res_scale * ffn_out
        x = self.layer_norm_ffn(x)

        return x, text_metrics

class FiLM(torch.nn.Module):
    def __init__(self, condition_dim, feature_channels, init_scale=0.5):
        """
        Args:
            condition_dim: Dimension of the global text embedding.
            feature_channels: Number of channels in the mel features.
            init_scale: Initial scaling factor.
        """
        super().__init__()
        # Enhanced FiLM conditioning with intermediate layers
        self.condition_net = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, condition_dim * 2),
            torch.nn.LayerNorm(condition_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )
        
        # Separate networks for gamma and beta with proper initialization
        self.gamma_fc = torch.nn.Linear(condition_dim * 2, feature_channels)
        self.beta_fc = torch.nn.Linear(condition_dim * 2, feature_channels)
        
        # Initialize with small values so initial conditioning is subtle
        torch.nn.init.xavier_normal_(self.gamma_fc.weight, gain=0.5)
        torch.nn.init.zeros_(self.gamma_fc.bias)
        torch.nn.init.xavier_normal_(self.beta_fc.weight, gain=0.5)
        torch.nn.init.zeros_(self.beta_fc.bias)
        
        # Use separate scale parameters with different defaults
        self.gamma_scale = torch.nn.Parameter(torch.tensor(init_scale))
        self.beta_scale = torch.nn.Parameter(torch.tensor(init_scale * 0.5))  # Beta starts smaller
        
        # Add a global scale parameter for overall conditioning strength
        self.global_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, cond):
        text_metrics = {} # for wandb logging

        # Debug input shapes
        print(f"[DEBUG] FiLM x shape: {x.shape}, cond shape: {cond.shape}")
        
        # Enhanced conditioning through MLP
        cond_features = self.condition_net(cond)
        
        # Get modulation parameters
        gamma_raw = self.gamma_fc(cond_features).unsqueeze(1)  # (B, 1, feature_channels)
        beta_raw = self.beta_fc(cond_features).unsqueeze(1)    # (B, 1, feature_channels)
        
        # Apply tanh to gamma to keep it in a reasonable range [-1, 1]
        # This prevents extreme scaling that could destabilize training
        gamma = torch.tanh(gamma_raw) * self.gamma_scale.clamp(min=0.1, max=1.0)
        
        # Don't constrain beta as much - allow it to shift the features
        beta = beta_raw * self.beta_scale.clamp(min=0.1, max=1.0)
        
        # Ensure gamma and beta have the right dimensions for broadcasting
        if x.dim() == 3:
            if gamma.shape[1] != x.shape[1]:
                # Expand to match the sequence length
                gamma = gamma.expand(-1, x.shape[1], -1)
                beta = beta.expand(-1, x.shape[1], -1)
                print(f"[DEBUG] Expanded shapes - gamma: {gamma.shape}, beta: {beta.shape}")
        
        # Calculate gamma centered at 1.0 for multiplicative scaling
        # This makes gamma=0 correspond to no change (x * 1.0)
        gamma_centered = 1.0 + gamma
        
        # Add these debug prints
        print(f"[DEBUG] FiLM gamma stats: min={gamma.min().item():.4f}, max={gamma.max().item():.4f}")
        print(f"[DEBUG] FiLM beta stats: min={beta.min().item():.4f}, max={beta.max().item():.4f}")
        
        # Get scales as float values to avoid tensor indexing issues
        global_scale = self.global_scale.clamp(min=0.5, max=2.0).item()
        
        # Apply FiLM conditioning: scale then shift
        # x' = global_scale * (gamma_centered * x + beta)
        result = global_scale * (gamma_centered * x + beta)
        
        print(f"[DEBUG] FiLM input magnitude: {x.abs().mean().item():.4f}")
        print(f"[DEBUG] FiLM output magnitude: {result.abs().mean().item():.4f}")
        print(f"[DEBUG] FiLM global scale: {global_scale:.4f}")

        # wandb logging
        text_metrics["film_gamma_min"] = gamma.min().item()
        text_metrics["film_gamma_max"] = gamma.max().item()
        text_metrics["film_beta_min"] = beta.min().item()
        text_metrics["film_beta_max"] = beta.max().item()
        text_metrics["film_input_magnitude"] = x.abs().mean().item()
        text_metrics["film_output_magnitude"] = result.abs().mean().item()
        text_metrics["film_global_scale"] = global_scale
        
        return result, text_metrics

class ConditionerNetwork(torch.nn.Module):
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
        output_channels=None,
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        text_encoder_config=None,  # This will be set to an instance of PLBERTTextEncoder's config.
        film_global_dim=256,       # Dimension for the global text embedding used in FiLM.
        cross_attention_dim=256    # Dimension for cross-attention.
    ):
        super().__init__()
        
        # Initialize training tracking variables
        self.current_epoch = 0  # Will be updated by parent module
        self.input_conv = cond_weight_norm(
            torch.nn.Conv1d(input_channels, n_channels, kernel_size=fb_kernel_size, padding="same"),
            use=use_weight_norm,
        )

        if output_channels is not None:
            self.output_conv = cond_weight_norm(
                torch.nn.Conv1d(n_channels, output_channels, kernel_size=fb_kernel_size, padding="same"),
                use=use_weight_norm,
            )
        else:
            self.output_conv = None

        total_ds = math.prod(rate_factors)
        total_channels = 2 ** len(rate_factors) * n_channels  # e.g., 512
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
            use_antialiasing=False,
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

        # Text Encoder & Fusion
        self.text_encoder = None
        self.film_global = None
        self.cross_attention = None
        self.mel_to_attn = None
        self.attn_to_mel = None

        if text_encoder_config is not None:
            # Instantiate PLBERT text encoder (which we defined in textencoder.py)
            self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
            print("[DEBUG] PLBERT TextEncoder instantiated:", self.text_encoder)

            # FiLM for global conditioning: projects global embedding onto mel feature channels.
            self.film_global = FiLM(condition_dim=film_global_dim, feature_channels=self.total_channels)
            # Projection: mel features (total_channels) → cross_attention_dim
            self.mel_to_attn = torch.nn.Linear(self.total_channels, cross_attention_dim)
            # Cross-attention block (for token-level fusion)
            # self.cross_attention = CrossAttentionBlock(hidden_dim=cross_attention_dim, num_heads=4)
            
            # Increase number of heads and add a second layer
            self.cross_attention = torch.nn.ModuleList([
                CrossAttentionBlock(hidden_dim=cross_attention_dim, num_heads=8),
                CrossAttentionBlock(hidden_dim=cross_attention_dim, num_heads=8)
            ])
            
            # Projection: cross_attention_dim → total_channels
            self.attn_to_mel = torch.nn.Linear(cross_attention_dim, self.total_channels)


            # Add these lines here to properly initialize the projection layers
            torch.nn.init.xavier_uniform_(self.mel_to_attn.weight)
            torch.nn.init.xavier_uniform_(self.attn_to_mel.weight)
            torch.nn.init.zeros_(self.mel_to_attn.bias) 
            torch.nn.init.zeros_(self.attn_to_mel.bias)
            
            self.text_direct_proj = torch.nn.Linear(film_global_dim, self.total_channels)
            torch.nn.init.xavier_uniform_(self.text_direct_proj.weight)
            torch.nn.init.zeros_(self.text_direct_proj.bias)

            self.post_text_norm = torch.nn.BatchNorm1d(total_channels)
            # Initialize with a positive value to work well with sigmoid in the blend formula
            # 1.0 will result in sigmoid(1.0) ≈ 0.73, giving blend_factor ≈ 0.77
            self.text_impact_factor = torch.nn.Parameter(torch.tensor(0.4)) #  1.0
            # Remove unused parameter
            # self.audio_bias = torch.nn.Parameter(torch.tensor(0.7))
            print("[DEBUG] Text conditioning components initialized.")
        else:
            print("[DEBUG] No TextEncoder provided; skipping text conditioning.")

    # def forward(self, x, x_wav=None, train=False, text=None):

    #     text_metrics = {} # for wandb logging

    #     n_samples = x.shape[-1]
    #     if x_wav is None:
    #         x_wav = x

    #     x_mel = self.input_mel(x_wav)  # (B, total_channels, T_mel)

    #     if self.text_encoder is not None and text is not None:
    #         # Store the original mel features before text conditioning
    #         x_mel_orig = x_mel.clone()
            
    #         # Find valid text inputs (non-empty)
    #         valid_indices = [i for i, t in enumerate(text) if t.strip()]

    #         if not valid_indices:
    #             # If all transcripts are empty, skip text conditioning
    #             print("[DEBUG] All transcripts are empty, skipping text conditioning")
    #         else:
    #             # Set text encoder to training mode when appropriate
    #             if train:
    #                 self.text_encoder.train()
    #             else:
    #                 self.text_encoder.eval()
                
    #             # Process only valid transcripts
    #             valid_text = [text[i] for i in valid_indices]
    #             global_emb, seq_emb = self.text_encoder(valid_text)

    #             print(f"[DEBUG] Global embedding stats: min={global_emb.min().item():.4f}, max={global_emb.max().item():.4f}, " 
    #                 f"mean={global_emb.mean().item():.4f}, std={global_emb.std().item():.4f}")
    #             print(f"[DEBUG] Sequence embedding shape: {seq_emb.shape}")

    #             # log for wandb
    #             text_metrics["global_emb_min"] = global_emb.min().item()
    #             text_metrics["global_emb_max"] = global_emb.max().item()
    #             text_metrics["global_emb_mean"] = global_emb.mean().item()
    #             text_metrics["global_emb_std"] = global_emb.std().item()

    #             # Create a copy of original mel features for residual connection
    #             x_mel_orig = x_mel.clone()
                
    #             # Apply global conditioning first:
    #             # -----------------------------
    #             # FiLM modulation (transpose to sequence-first for FiLM)
    #             x_mel_t = x_mel.transpose(1, 2)  # (B, T_mel, total_channels)
                
    #             # Apply FiLM modulation with enhanced conditioning
    #             x_mel_t, film_metrics = self.film_global(x_mel_t, global_emb)  # (B, T_mel, total_channels)
    #             text_metrics.update(film_metrics)  # Merge metrics dictionaries
                
    #             # Apply token-level conditioning:
    #             # -----------------------------
    #             # Project mel features to cross-attention dimension
    #             x_mel_attn = self.mel_to_attn(x_mel_t)  # (B, T_mel, cross_attention_dim)
                
    #             # Apply progressive cross-attention (sequence of layers)
    #             for i, attn_layer in enumerate(self.cross_attention):
    #                 x_mel_attn, layer_metrics = attn_layer(x_mel_attn, seq_emb)
    #                 # Add layer index to metrics for better tracking
    #                 prefixed_metrics = {f"layer{i}_{k}": v for k, v in layer_metrics.items()}
    #                 text_metrics.update(prefixed_metrics)
                
    #             # Project back to original mel dimension
    #             x_mel_t_ca = self.attn_to_mel(x_mel_attn)  # (B, T_mel, total_channels)
                
    #             # Convert both representations back to channel-first
    #             x_mel_film = x_mel_t.transpose(1, 2)  # FiLM output (B, C, T)
    #             x_mel_ca = x_mel_t_ca.transpose(1, 2)   # Cross-attention output (B, C, T)
                
    #             # Simple adaptive balancing of global vs token-level conditioning
    #             # Start with more global conditioning (70% FiLM, 30% cross-attention)
                
    #             # Add an epoch-dependent scaling (UPDATED)
    #             # This ramps up the blend factor over the first 10 epochs
    #             if hasattr(self, 'current_epoch'):
    #                 current_epoch = self.current_epoch
    #             elif hasattr(self, '_parameters') and 'text_impact_factor' in self._parameters:
    #                 # Try to access from parent module if available
    #                 if hasattr(self._parameters['text_impact_factor'], '_cdata'):
    #                     parent_module = self._parameters['text_impact_factor']._cdata.obj
    #                     current_epoch = getattr(parent_module, 'current_epoch', 0)
    #                 else:
    #                     current_epoch = 0
    #             else:
    #                 current_epoch = 0

                
    #             training_progress = min(1.0, current_epoch / 10)  # Ramp up over 10 epochs
                
    #             # This helps stabilize early training
    #             film_weight = 0.5 # updated
    #             ca_weight = 0.5 # updated
                
    #             # Optionally, make this progressive too
    #             if training_progress < 0.3:
    #                 # In early training, rely slightly more on global cues
    #                 film_weight = 0.6
    #                 ca_weight = 0.4
                
    #             # Combine FiLM and cross-attention results
    #             x_mel_conditioned = film_weight * x_mel_film + ca_weight * x_mel_ca
                
    #             # Before blending with the impact factor
    #             print(f"[DEBUG] Before conditioning - Mel features magnitude: {x_mel_orig.abs().mean().item():.4f}")
    #             print(f"[DEBUG] Conditioned features magnitude: {x_mel_conditioned.abs().mean().item():.4f}")
                
    #             # Use sigmoid for smooth blending factor with a higher baseline
    #             # This maps self.text_impact_factor to range 0.4-0.9
    #             # Starts higher for more text impact
                
    #             # raw_factor = torch.sigmoid(self.text_impact_factor)
    #             # blend_factor = 0.4 + 0.5 * raw_factor
    #             # print(f"[DEBUG] Blend factor: {blend_factor.item():.4f} (0.4-0.9 range)")
                
                
                
                
    #             # Add an epoch-dependent scaling
    #             # Get the current epoch from the current state
    #             # Default to 0 if not available
    #             current_epoch = 0  # Default value
    #             training_progress = min(1.0, current_epoch / 10)  # Ramp up over 10 epochs

    #             # You can also just use a fixed low value for now
    #             max_blend = 0.7  # Maximum blend factor
    #             min_blend = 0.2  # Minimum starting blend
    #             blend_factor = min_blend + (max_blend - min_blend) * training_progress

    #             # Still use the impact factor for fine-tuning
    #             raw_factor = torch.sigmoid(self.text_impact_factor) 
    #             blend_factor = 0.2 + 0.1 * raw_factor  # Keep it in a lower range: 0.2-0.3

    #             print(f"[DEBUG] Fixed lower blend factor: {blend_factor.item():.4f}")
                                
    #             # Create a residual mask to apply text features with varying intensity
    #             # This applies more conditioning to the middle frequency range where speech formants are
    #             # Attenuates conditioning at very low and very high frequencies
    #             # This is a simple frequency-based prior to help the model use text where it matters most
    #             freq_profile = torch.ones_like(x_mel_orig)
    #             _, C, _ = freq_profile.shape
    #             for c in range(C):
    #                 # Create a bow shape with peak in the middle frequencies
    #                 # Simple heuristic: 1.0 in middle decreasing to 0.7 at edges
    #                 rel_pos = abs(c / C - 0.5) * 2.0  # 0 (middle) to 1 (edges)
    #                 freq_scale = 1.0 - 0.3 * rel_pos  # 1.0 in middle to 0.7 at edges
    #                 freq_profile[:, c, :] *= freq_scale
                
    #             # Combine original features with conditioned features
    #             # Apply frequency-dependent blending
    #             freq_aware_blend = blend_factor * freq_profile
    #             x_mel = (1.0 - freq_aware_blend) * x_mel_orig + freq_aware_blend * x_mel_conditioned
                
    #             # Simple magnitude preservation to avoid signal weakening
    #             avg_orig = x_mel_orig.abs().mean()
    #             avg_new = x_mel.abs().mean()
    #             scale = avg_orig / avg_new.clamp(min=1e-8)
                
    #             # Apply scaling to maintain similar magnitude but preserve text features
    #             x_mel = x_mel * scale
                
    #             # Log all factors
    #             print(f"[DEBUG] FiLM/CA ratio: {film_weight:.2f}/{ca_weight:.2f}")
    #             print(f"[DEBUG] Scaling factor: {scale.item():.4f}")
    #             text_metrics["film_weight"] = film_weight
    #             text_metrics["ca_weight"] = ca_weight
    #             text_metrics["scaling_factor"] = scale.item()
                
    #             print(f"[DEBUG] Text impact factor: {self.text_impact_factor.item()}")
    #             print(f"[DEBUG] After blending and normalization - Mel features magnitude: {x_mel.abs().mean().item()}")
    #             print(f"[DEBUG] Feature difference magnitude: {(x_mel - x_mel_orig).abs().mean().item()}")

    #             # Feature metrics
    #             text_metrics["mel_features_before"] = x_mel_orig.abs().mean().item()
    #             text_metrics["mel_features_conditioned"] = x_mel_conditioned.abs().mean().item()
    #             text_metrics["blend_factor"] = blend_factor.item()
    #             text_metrics["text_impact_factor"] = self.text_impact_factor.item()
    #             text_metrics["mel_features_after"] = x_mel.abs().mean().item()
    #             text_metrics["feature_difference"] = (x_mel - x_mel_orig).abs().mean().item()
                

    #     if self.precoding:
    #         x = self.precoding(x)

    #     x = self.input_conv(x)
    #     h, lengths = self.encoder(x, x_mel)
    #     y_hat, conditions = self.decoder(h, lengths)

    #     if self.output_conv is not None:
    #         y_hat = self.output_conv(y_hat)
    #     if self.precoding:
    #         y_hat = self.precoding.inv(y_hat)

    #     y_hat = torch.nn.functional.pad(y_hat, (0, n_samples - y_hat.shape[-1]))

    #     if train:
    #         return conditions, y_hat, h, text_metrics
    #     else:
    #         return conditions if not self.text_encoder or text is None else (conditions, text_metrics)
    
    
    def forward(self, x, x_wav=None, train=False, text=None, current_epoch=0):

        text_metrics = {} # for wandb logging
        
        # Store text embeddings for external use
        self.text_global_embedding = None
        self.text_seq_embedding = None

        n_samples = x.shape[-1]
        if x_wav is None:
            x_wav = x

        x_mel = self.input_mel(x_wav)  # (B, total_channels, T_mel)

        if self.text_encoder is not None and text is not None:
            # Store the original mel features before text conditioning
            x_mel_orig = x_mel.clone()
            
            # Find valid text inputs (non-empty)
            valid_indices = [i for i, t in enumerate(text) if t.strip()]

            if not valid_indices:
                # If all transcripts are empty, skip text conditioning
                print("[DEBUG] All transcripts are empty, skipping text conditioning")
            else:
                # Set text encoder to training mode when appropriate
                if train:
                    self.text_encoder.train()
                else:
                    self.text_encoder.eval()
                
                # Process only valid transcripts
                valid_text = [text[i] for i in valid_indices]
                global_emb, seq_emb = self.text_encoder(valid_text)
                
                # Store embeddings for use by score network
                self.text_global_embedding = global_emb
                self.text_seq_embedding = seq_emb

                print(f"[DEBUG] Global embedding stats: min={global_emb.min().item():.4f}, max={global_emb.max().item():.4f}, " 
                    f"mean={global_emb.mean().item():.4f}, std={global_emb.std().item():.4f}")
                print(f"[DEBUG] Sequence embedding shape: {seq_emb.shape}")

                # log for wandb
                text_metrics["global_emb_min"] = global_emb.min().item()
                text_metrics["global_emb_max"] = global_emb.max().item()
                text_metrics["global_emb_mean"] = global_emb.mean().item()
                text_metrics["global_emb_std"] = global_emb.std().item()

                # Create a copy of original mel features for residual connection
                x_mel_orig = x_mel.clone()
                
                # Apply global conditioning first:
                # -----------------------------
                # FiLM modulation (transpose to sequence-first for FiLM)
                x_mel_t = x_mel.transpose(1, 2)  # (B, T_mel, total_channels)
                
                # Apply FiLM modulation with enhanced conditioning
                x_mel_t, film_metrics = self.film_global(x_mel_t, global_emb)  # (B, T_mel, total_channels)
                text_metrics.update(film_metrics)  # Merge metrics dictionaries
                
                # Apply token-level conditioning:
                # -----------------------------
                # Project mel features to cross-attention dimension
                x_mel_attn = self.mel_to_attn(x_mel_t)  # (B, T_mel, cross_attention_dim)
                
                # Apply progressive cross-attention (sequence of layers)
                for i, attn_layer in enumerate(self.cross_attention):
                    x_mel_attn, layer_metrics = attn_layer(x_mel_attn, seq_emb)
                    # Add layer index to metrics for better tracking
                    prefixed_metrics = {f"layer{i}_{k}": v for k, v in layer_metrics.items()}
                    text_metrics.update(prefixed_metrics)
                
                # Project back to original mel dimension
                x_mel_t_ca = self.attn_to_mel(x_mel_attn)  # (B, T_mel, total_channels)
                
                # Convert both representations back to channel-first
                x_mel_film = x_mel_t.transpose(1, 2)  # FiLM output (B, C, T)
                x_mel_ca = x_mel_t_ca.transpose(1, 2)   # Cross-attention output (B, C, T)
                
                # # Balance global vs token-level conditioning based on training stage
                # film_weight = 0.5
                # ca_weight = 0.5
                
                # # In early training, rely more on global cues
                # if train and current_epoch < 3:
                #     film_weight = 0.6
                #     ca_weight = 0.4
                
                # With adaptive ratio:
                if current_epoch < 10:
                    # Start with more global FiLM conditioning
                    film_weight = 0.7
                    ca_weight = 0.3
                elif current_epoch < 20:
                    # Gradually shift to more token-level conditioning
                    film_weight = 0.6
                    ca_weight = 0.4
                else:
                    # More emphasis on cross-attention for fine details
                    film_weight = 0.5
                    ca_weight = 0.5
                                
                    
                # Combine FiLM and cross-attention results
                x_mel_conditioned = film_weight * x_mel_film + ca_weight * x_mel_ca
                
                
                # TRAINING-AWARE BLEND FACTOR CALCULATION
                
                # Try to get epoch info from global variables in universe_gan_NS3
                try:
                    # Import the module that has our global variable
                    from ..universe.universe_gan_NS3 import CURRENT_EPOCH, GLOBAL_STEP
                    current_epoch = CURRENT_EPOCH
                    global_step = GLOBAL_STEP
                    print(f"[EPOCH CHECK] Got epoch from universe_gan_NS3: {current_epoch}")
                except (ImportError, AttributeError):
                    # Fallback: Start with zero epoch
                    current_epoch = 0
                    global_step = 0
                    print("[EPOCH CHECK] Warning: Could not access epoch from universe_gan_NS3")
                    
                    # Alternative: Use global_step if available to estimate epoch
                    global_step = getattr(self, 'global_step', 0)
                    if global_step > 0:
                        # Assume ~1000 steps per epoch
                        current_epoch = int(global_step / 1000)
                        print(f"[EPOCH CHECK] Estimated epoch from steps: {current_epoch}")
                
                # Start with extremely low text influence and gradually ramp up over 20 epochs
                training_progress = min(1.0, current_epoch / 20.0)  # Ramp up over 20 epochs
                
                
                
                # Add a small direct contribution from text embeddings
                if training_progress > 0.2:  # Only enable after some initial training
                    # Project global embedding to mel dimension
                    direct_proj = self.text_direct_proj(global_emb)  # Add this layer to __init__
                    direct_proj = direct_proj.unsqueeze(-1).expand(-1, -1, x_mel_conditioned.shape[-1])
                    
                    # Add small direct contribution 
                    direct_scale = 0.1 * min(1.0, (training_progress - 0.2) / 0.3)
                    x_mel_conditioned = x_mel_conditioned + direct_scale * direct_proj
                                
                                
                
                # Before blending with the impact factor
                mel_orig_mag = x_mel_orig.abs().mean().item()
                mel_cond_mag = x_mel_conditioned.abs().mean().item()
                print(f"[MEL DEBUG] Before conditioning - Mel features magnitude: {mel_orig_mag:.4f}")
                print(f"[MEL DEBUG] Conditioned features magnitude: {mel_cond_mag:.4f}")
                print(f"[MEL DEBUG] Conditioning relative impact: {(mel_cond_mag - mel_orig_mag) / mel_orig_mag:.4f}")
                
                
                # Apply a curve so that growth is slower at start and faster later
                # using a cubic curve: training_progress^3
                training_curve = training_progress ** 3
                
                # Very small base value (0.02) that increases to 0.3 over 20 epochs
                # Start extremely small to avoid overwhelming the model
                base_blend = 0.15 * training_curve  # Starts at 0.0 and grows to 0.02
                
                # Add a small amount from learnable parameter
                raw_factor = torch.sigmoid(self.text_impact_factor)
                learnable_contribution = 0.25 * raw_factor * training_curve  # Max additional 0.03
                
                # Final blend factor
                blend_factor = base_blend + learnable_contribution  # Range: ~0.001 to 0.05 over training
                
                # Add training progress to debug info
                print(f"[MEL DEBUG] Training progress: {training_progress:.4f} (epoch {current_epoch}/20)")
                print(f"[MEL DEBUG] Training curve: {training_curve:.4f} (cubic progression)")
                
                print(f"[MEL DEBUG] Blend factor: {blend_factor.item():.4f} (text impact strength)")
                print(f"[MEL DEBUG] Raw text impact param: {self.text_impact_factor.item():.4f} (learnable)")
                
                # Create a residual mask to apply text features with varying intensity
                # This applies more conditioning to the middle frequency range where speech formants are
                # Attenuates conditioning at very low and very high frequencies
                freq_profile = torch.ones_like(x_mel_orig)
                _, C, _ = freq_profile.shape
                for c in range(C):
                    # Create a bow shape with peak in the middle frequencies
                    # Simple heuristic: 1.0 in middle decreasing to 0.7 at edges
                    rel_pos = abs(c / C - 0.5) * 2.0  # 0 (middle) to 1 (edges)
                    freq_scale = 1.0 - 0.3 * rel_pos  # 1.0 in middle to 0.7 at edges
                    freq_profile[:, c, :] *= freq_scale
                
                # Combine original features with conditioned features
                # Apply frequency-dependent blending
                freq_aware_blend = blend_factor * freq_profile
                x_mel = (1.0 - freq_aware_blend) * x_mel_orig + freq_aware_blend * x_mel_conditioned
                
                # Simple magnitude preservation to avoid signal weakening
                avg_orig = x_mel_orig.abs().mean()
                avg_new = x_mel.abs().mean()
                scale = avg_orig / avg_new.clamp(min=1e-8)
                
                # Apply scaling to maintain similar magnitude but preserve text features
                x_mel = x_mel * scale
                
                # Calculate final impact metrics
                final_mag = x_mel.abs().mean().item()
                diff_mag = (x_mel - x_mel_orig).abs().mean().item()
                rel_impact = diff_mag / mel_orig_mag
                
                # Log all factors with clearer labels
                print(f"[MEL DEBUG] FiLM/CA ratio: {film_weight:.2f}/{ca_weight:.2f}")
                print(f"[MEL DEBUG] Magnitude scaling factor: {scale.item():.4f}")
                print(f"[MEL DEBUG] After blending - Mel features magnitude: {final_mag:.4f}")
                print(f"[MEL DEBUG] Feature difference magnitude: {diff_mag:.4f}")
                print(f"[MEL DEBUG] Relative impact of text: {rel_impact:.4f} (higher = stronger)")
                
                # Extended metrics for wandb logging
                text_metrics["mel_film_weight"] = film_weight
                text_metrics["mel_ca_weight"] = ca_weight
                text_metrics["mel_scaling_factor"] = scale.item()
                text_metrics["mel_features_before"] = mel_orig_mag
                text_metrics["mel_features_conditioned"] = mel_cond_mag
                text_metrics["mel_blend_factor"] = blend_factor.item()
                text_metrics["mel_text_impact_param"] = self.text_impact_factor.item()
                text_metrics["mel_features_after"] = final_mag
                text_metrics["mel_feature_difference"] = diff_mag
                text_metrics["mel_relative_impact"] = rel_impact
                
                # Add frequency profile metrics
                text_metrics["freq_profile_center"] = 1.0
                text_metrics["freq_profile_edge"] = 0.7  # 1.0 - 0.3
                text_metrics["freq_aware_blend_max"] = blend_factor.item() # Center freq max blend
                
                # Add training progression metrics
                text_metrics["training_progress"] = training_progress
                text_metrics["training_curve"] = training_curve
                text_metrics["current_epoch"] = current_epoch

        if self.precoding:
            x = self.precoding(x)

        x = self.input_conv(x)
        h, lengths = self.encoder(x, x_mel)
        y_hat, conditions = self.decoder(h, lengths)

        if self.output_conv is not None:
            y_hat = self.output_conv(y_hat)
        if self.precoding:
            y_hat = self.precoding.inv(y_hat)

        y_hat = torch.nn.functional.pad(y_hat, (0, n_samples - y_hat.shape[-1]))

        if train:
            return conditions, y_hat, h, text_metrics, self.text_global_embedding
        else:
            if not self.text_encoder or text is None:
                return conditions
            else:
                return conditions, text_metrics, self.text_global_embedding