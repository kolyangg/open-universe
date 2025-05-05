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

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        """
        hidden_dim: dimension of mel/text embeddings
        num_heads: number of attention heads
        """
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        
        # Optional: A small feed-forward network after attention
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm_ffn = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, cond, x_mask=None, cond_mask=None):
        """
        x: (B, T_x, hidden_dim)        # e.g. FiLM-modulated mel features
        cond: (B, T_cond, hidden_dim)  # text embeddings
        x_mask, cond_mask: optional attention masks
        """
        # Cross-attention: query = x, key & value = cond
        attn_out, _ = self.cross_attn(x, cond, cond, 
                                      key_padding_mask=cond_mask,
                                      need_weights=False)
        x = x + attn_out
        x = self.layer_norm(x)

        # Optional feed-forward
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.layer_norm_ffn(x)

        return x



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
        # optional, if specified, an extra conv. layer is used as adapter
        # for the output signal estimat y_est
        output_channels=None,
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        text_encoder_config=None,  # новый параметр для текстового энкодера
        film_global_dim=256,    # e.g. dimension for global text embedding
        cross_attention_dim=256 # dimension for cross-attn embeddings
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
        total_channels = 2 ** len(rate_factors) * n_channels
        self.total_channels = total_channels  # store for later use

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

        ##### NEW TEXT ENCODER #####
        # self.n_mels = n_mels # TEMP

        # text_encoder_config = None # TEMP!!!! 
        # self.text_encoder = None

        # if text_encoder_config is not None:
        #     self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
        #     # Project the text encoder's hidden dimension to match total_channels (e.g., 512)
           
        #     self.film = FiLM(
        #         condition_dim=text_encoder_config.hidden_dim,
        #         feature_channels=self.total_channels
        #     )

        #     print("[DEBUG] TextEncoder instantiated:", self.text_encoder)
        # else:
        #     self.text_encoder = None
        #     print("[DEBUG] No TextEncoder")

         # We'll add two projection layers to convert between self.total_channels and cross_attention_dim
        self.mel_to_attn = None
        self.attn_to_mel = None

        if text_encoder_config is not None:
            # 1) Instantiate your text encoder
            self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
            print("[DEBUG] TextEncoder instantiated:", self.text_encoder)

            # 2) FiLM for global conditioning
            #    Suppose the text encoder can output two things:
            #    (a) a global vector (e.g., speaker/style) for FiLM
            #    (b) a sequence of embeddings for cross-attention
            self.film_global = FiLM(
                condition_dim=film_global_dim,
                feature_channels=self.total_channels
            )

            # Projection from mel feature dimension to cross-attention dimension
            self.mel_to_attn = torch.nn.Linear(self.total_channels, cross_attention_dim)

            # 3) CrossAttention block for text <-> mel
            self.cross_attention = CrossAttentionBlock(
                hidden_dim=cross_attention_dim,  # must match your text seq embedding dimension
                num_heads=4
            )

            # Projection from cross-attention dimension back to mel feature dimension
            self.attn_to_mel = torch.nn.Linear(cross_attention_dim, self.total_channels)
        else:
            print("[DEBUG] No TextEncoder provided; skipping text conditioning.")
        
        ##### NEW TEXT ENCODER #####

    def forward(self, x, x_wav=None, train=False, text=None):
       
        n_samples = x.shape[-1]

        if x_wav is None:
            x_wav = x

        x_mel = self.input_mel(x_wav)


        ##### NEW TEXT ENCODER #####
        # if self.text_encoder is not None and text is not None:
        #     # Obtain text embedding of shape (B, hidden_dim)
        #     text_emb = self.text_encoder(text)
            
        #     # Apply FiLM fusion: modulate x_mel with text conditions
        #     x_mel = self.film(x_mel, text_emb)
        #     # x_mel = x_mel # TEMP
            
        #     # Debug prints for verification
        #     # print(f"Debug: x_mel shape after FiLM: {x_mel.shape}")
        #     # assert x_mel.shape == text_emb.shape, f"Shape mismatch: x_mel {x_mel.shape} vs text_emb {text_emb.shape}"

        #     # print("[DEBUG] Text features integrated into mel: shape", x_mel.shape)
        # ##### NEW TEXT ENCODER #####
        # # else:
        #     # print("[DEBUG] No Text Features in fwd pass")

        ### NEW VERSION WITH CROSS ATTENTION ###
        if self.text_encoder is not None and text is not None:
            # Example: text_encoder outputs (global_emb, seq_emb)
            #   global_emb: (B, film_global_dim)
            #   seq_emb: (B, T_text, cross_attention_dim)
            global_emb, seq_emb = self.text_encoder(text)

            # # 2a) FiLM for global conditioning
            # # We need x_mel in shape (B, T_mel, total_channels)
            # x_mel_t = x_mel.transpose(1, 2)  # now (B, T_mel, total_channels)
            # x_mel_t = self.film_global(x_mel_t, global_emb)  # apply FiLM
            # # shape remains (B, T_mel, total_channels)

            # # 2b) Cross-Attention
            # # We assume x_mel_t and seq_emb have the same embedding dimension
            # # x_mel_t: (B, T_mel, cross_attention_dim)
            # # seq_emb: (B, T_text, cross_attention_dim)
            # x_mel_t = self.cross_attention(x_mel_t, seq_emb)

            # # transpose back to (B, total_channels, T_mel)
            # x_mel = x_mel_t.transpose(1, 2)

            # 2a) FiLM for global conditioning
            x_mel_t = x_mel.transpose(1, 2)  # (B, T_mel, total_channels)
            x_mel_t = self.film_global(x_mel_t, global_emb)  # (B, T_mel, total_channels)

            # 2b) Project mel features to cross-attention dimension (e.g., 256)
            x_mel_attn = self.mel_to_attn(x_mel_t)  # (B, T_mel, cross_attention_dim)

            # Apply cross-attention: now x_mel_attn and seq_emb are both 256-dim
            x_mel_attn = self.cross_attention(x_mel_attn, seq_emb)  # (B, T_mel, cross_attention_dim)

            # Project back to original mel feature dimension
            x_mel_t = self.attn_to_mel(x_mel_attn)  # (B, T_mel, total_channels)

            # Transpose back to (B, total_channels, T_mel)
            x_mel = x_mel_t.transpose(1, 2)




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
            return conditions, y_hat, h
        else:
            return conditions
        

class FiLM(torch.nn.Module): # updated
    def __init__(self, condition_dim, feature_channels):
        """
        condition_dim: Dimension of text embedding (from text encoder)
        feature_channels: Number of channels in the mel features
        """
        super().__init__()
        self.gamma_fc = torch.nn.Linear(condition_dim, feature_channels)
        self.beta_fc = torch.nn.Linear(condition_dim, feature_channels)

    def forward(self, x, cond):
        """
        x: (B, T_x, feature_channels)   # Mel features (transposed)
        cond: (B, condition_dim)        # Possibly a single global vector (e.g., speaker/style)
        """
        gamma = self.gamma_fc(cond).unsqueeze(1)  # (B, 1, feature_channels)
        beta = self.beta_fc(cond).unsqueeze(1)    # (B, 1, feature_channels)
        return gamma * x + beta
