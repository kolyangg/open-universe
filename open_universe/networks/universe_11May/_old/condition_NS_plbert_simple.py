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
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm_ffn = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, cond, x_mask=None, cond_mask=None):

        text_metrics = {} # for wandb logging

        attn_out, attn_weights = self.cross_attn(x, cond, cond, 
                                      key_padding_mask=cond_mask,
                                      need_weights=True)
        
        attn_weights_mean_per_head = attn_weights.mean(dim=1)  # Average across sequence
        max_attentions = attn_weights.max(dim=-1)[0]  # Max attention per position
        print(f"[DEBUG] Attention focus: {max_attentions.mean().item():.4f} (higher = more focused)")
        print(f"[DEBUG] Attention weights stats: min={attn_weights.min().item()}, " 
            f"max={attn_weights.max().item()}, mean={attn_weights.mean().item()}")
        
        text_metrics["attention_focus"] = max_attentions.mean().item()
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
                
        
        x = x + attn_out
        x = self.layer_norm(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.layer_norm_ffn(x)

        return x, text_metrics


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
        text_encoder_config=None,
        cross_attention_dim=256 
    ):
        super().__init__()
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

        # Text processing components
        self.text_encoder = None
        self.cross_attention = None
        self.mel_to_attn = None
        self.attn_to_mel = None

        if text_encoder_config is not None:
            # Instantiate PLBERT text encoder
            self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
            print("[DEBUG] PLBERT TextEncoder instantiated:", self.text_encoder)

            # Project mel features to cross-attention dimension
            self.mel_to_attn = torch.nn.Linear(self.total_channels, cross_attention_dim)
            
            # Cross-attention block for directly connecting text features with mel
            self.cross_attention = CrossAttentionBlock(hidden_dim=cross_attention_dim, num_heads=4)
            
            # Project back to original mel dimension
            self.attn_to_mel = torch.nn.Linear(cross_attention_dim, self.total_channels)

            # Initialize projection layers properly
            torch.nn.init.xavier_uniform_(self.mel_to_attn.weight)
            torch.nn.init.xavier_uniform_(self.attn_to_mel.weight)
            torch.nn.init.zeros_(self.mel_to_attn.bias) 
            torch.nn.init.zeros_(self.attn_to_mel.bias)

            # Hard cap on text impact to prevent over-reliance
            self.text_impact_factor = torch.nn.Parameter(torch.tensor(0.2))
            
            # Add layer normalization for maintaining feature distributions
            self.layer_norm_mel = torch.nn.LayerNorm(self.total_channels)
            
            print("[DEBUG] Text conditioning components initialized.")
        else:
            print("[DEBUG] No TextEncoder provided; skipping text conditioning.")

    def forward(self, x, x_wav=None, train=False, text=None):
        text_metrics = {}  # for wandb logging

        n_samples = x.shape[-1]
        if x_wav is None:
            x_wav = x

        x_mel = self.input_mel(x_wav)  # (B, total_channels, T_mel)

        if self.text_encoder is not None and text is not None:
            # Store the original mel features before text conditioning
            x_mel_orig = x_mel.clone()

            valid_indices = [i for i, t in enumerate(text) if t.strip()]

            if not valid_indices:
                print("[DEBUG] All transcripts are empty, skipping text conditioning")
            else:
                # Process only valid transcripts
                valid_text = [text[i] for i in valid_indices]
                
                # Get text embeddings from PL-BERT (we only use sequence embeddings now)
                _, seq_emb = self.text_encoder(valid_text)

                # Log embedding stats
                print(f"[DEBUG] Sequence embedding shape: {seq_emb.shape}")
                
                # Prepare mel features for cross-attention
                x_mel_t = x_mel.transpose(1, 2)  # (B, T_mel, total_channels)
                
                # Project mel features to attention dimension
                x_mel_attn = self.mel_to_attn(x_mel_t)  # (B, T_mel, cross_attention_dim)
                
                # Apply cross-attention directly between mel features and text sequence
                x_mel_attn, attn_metrics = self.cross_attention(x_mel_attn, seq_emb)
                text_metrics.update(attn_metrics)
                
                # Project back to original dimension
                x_mel_t = self.attn_to_mel(x_mel_attn)  # (B, T_mel, total_channels)
                
                # Apply layer normalization to stabilize features
                x_mel_t = self.layer_norm_mel(x_mel_t)
                
                # Convert back to channel-first format
                x_mel_conditioned = x_mel_t.transpose(1, 2)  # (B, total_channels, T_mel)
                
                # Before blending
                print(f"[DEBUG] Before conditioning - Mel features magnitude: {x_mel_orig.abs().mean().item()}")
                print(f"[DEBUG] Conditioned features magnitude: {x_mel_conditioned.abs().mean().item()}")
                
                # Constrain the text impact factor
                impact = torch.clamp(self.text_impact_factor, 0.0, 0.5) # 0.3 looks to low influence
                blend_factor = torch.sigmoid(impact)
                print(f"[DEBUG] Blend factor (should be 0-1): {blend_factor.item()}")
                
                # Blend original and conditioned features with controlled impact
                x_mel = (1.0 - blend_factor) * x_mel_orig + blend_factor * x_mel_conditioned
                
                # Ensure magnitude is preserved with simple normalization
                magnitude_ratio = (x_mel_orig.norm() / x_mel.norm()).clamp(min=0.5, max=2.0)
                x_mel = x_mel * magnitude_ratio
                
                # Log metrics
                print(f"[DEBUG] Text impact factor: {impact.item()}")
                print(f"[DEBUG] After blending - Mel features magnitude: {x_mel.abs().mean().item()}")
                print(f"[DEBUG] Feature difference magnitude: {(x_mel - x_mel_orig).abs().mean().item()}")
                
                # Store metrics for wandb
                text_metrics["mel_features_before"] = x_mel_orig.abs().mean().item()
                text_metrics["mel_features_conditioned"] = x_mel_conditioned.abs().mean().item()
                text_metrics["blend_factor"] = blend_factor.item()
                text_metrics["text_impact_factor"] = impact.item() 
                text_metrics["mel_features_after"] = x_mel.abs().mean().item()
                text_metrics["feature_difference"] = (x_mel - x_mel_orig).abs().mean().item()

        # Process through the rest of the model
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
            return conditions, y_hat, h, text_metrics
        else:
            return conditions if not self.text_encoder or text is None else (conditions, text_metrics)