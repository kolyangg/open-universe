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

Modified to move cross-attention after the GRU block and add debug logs to verify text-phonetic connections.
"""
import math
import torch
import torchaudio
from hydra.utils import instantiate

try:
    from .blocks import (
        BinomialAntiAlias,
        ConvBlock,
        PReLU_Conv,
        cond_weight_norm,
    )
except ImportError:
    from blocks import (
        BinomialAntiAlias,
        ConvBlock,
        PReLU_Conv,
        cond_weight_norm,
    )


###############################################################################
# Utility function for strided conv creation
###############################################################################
def make_st_convs(
    ds_factors,
    input_channels,
    num_layers=None,
    use_weight_norm=False,
    use_antialiasing=False,
):
    """
    Creates a list of 1D convolutions that adjust the rate (stride) and channels
    at each downsample stage. Used in the "st_convs" within the ConditionerEncoder.
    """
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
            i_chan = input_channels * 2 ** i
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


###############################################################################
# Mel spectrogram adapter
###############################################################################
class MelAdapter(torch.nn.Module):
    """
    Computes a mel-spectrogram from the input waveform, then applies a small
    convolutional block. Used as part of the multi-scale conditioner features.
    """
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

        # Figure out padding so that we get a clean number of frames
        pad_tot = n_fft - ds_factor
        self.pad_left, self.pad_right = pad_tot // 2, pad_tot - pad_tot // 2

    def compute_mel_spec(self, x):
        # Pad the waveform to align with fft/hop
        r = x.shape[-1] % self.ds_factor
        if r != 0:
            pad = self.ds_factor - r
        else:
            pad = 0

        x = torch.nn.functional.pad(x, (self.pad_left, pad + self.pad_right))
        x = self.mel_spec(x)  # -> (B, n_mels, frames)
        x = x.squeeze(1)

        # Global normalization of the mel-spectrogram
        norm = (x ** 2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).sqrt()
        x = x / norm.clamp(min=1e-5)

        return x

    def forward(self, x):
        x = self.compute_mel_spec(x)
        x = self.conv(x)
        x, *_ = self.conv_block(x)
        return x


###############################################################################
# Optional FiLM block for global text conditioning
###############################################################################
class FiLM(torch.nn.Module):
    """
    FiLM modulates the feature tensor x using a global conditioning vector.
    x' = scale * ( (1 + gamma) * x + beta ), with gamma, beta derived from the condition.
    """
    def __init__(self, condition_dim, feature_channels, init_scale=0.5):
        super().__init__()
        self.condition_net = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, condition_dim * 2),
            torch.nn.LayerNorm(condition_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1)
        )
        self.gamma_fc = torch.nn.Linear(condition_dim * 2, feature_channels)
        self.beta_fc = torch.nn.Linear(condition_dim * 2, feature_channels)

        # Initialize
        torch.nn.init.xavier_normal_(self.gamma_fc.weight, gain=0.5)
        torch.nn.init.zeros_(self.gamma_fc.bias)
        torch.nn.init.xavier_normal_(self.beta_fc.weight, gain=0.5)
        torch.nn.init.zeros_(self.beta_fc.bias)

        self.gamma_scale = torch.nn.Parameter(torch.tensor(init_scale))
        self.beta_scale = torch.nn.Parameter(torch.tensor(init_scale * 0.5))
        self.global_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, cond):
        """
        x:    (B, T, C)
        cond: (B, condition_dim)
        """
        cond_features = self.condition_net(cond)

        # Get gamma/beta
        gamma_raw = self.gamma_fc(cond_features).unsqueeze(1)  # (B, 1, C)
        beta_raw = self.beta_fc(cond_features).unsqueeze(1)   # (B, 1, C)

        gamma = torch.tanh(gamma_raw) * self.gamma_scale.clamp(min=0.1, max=1.0)
        beta = beta_raw * self.beta_scale.clamp(min=0.1, max=1.0)

        if x.dim() == 3:
            B, T, C = x.shape
            if gamma.shape[1] != T:
                gamma = gamma.expand(-1, T, -1)
                beta  = beta.expand(-1, T, -1)

        # (1 + gamma) scale
        gamma_centered = 1.0 + gamma
        scale = self.global_scale.clamp(0.5, 2.0)

        # Return debug metrics
        out = scale * (gamma_centered * x + beta)
        film_debug = {
            "film_gamma_min": gamma.min().item(),
            "film_gamma_max": gamma.max().item(),
            "film_scale": scale.item()
        }
        return out, film_debug


###############################################################################
# CrossAttentionBlock for token-level text conditioning
###############################################################################
class CrossAttentionBlock(torch.nn.Module):
    """
    Cross-attention block to fuse a 'query' sequence with a 'cond' (key/value) sequence.
    Uses multi-head attention and optional residual/FFN layers.
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.cross_attn = torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.pre_attn_norm = torch.nn.LayerNorm(hidden_dim)
        self.pre_cond_norm = torch.nn.LayerNorm(hidden_dim)

        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.layer_norm_ffn = torch.nn.LayerNorm(hidden_dim)

        # A temperature (scale) parameter for controlling attention sharpness
        self.attention_temperature = torch.nn.Parameter(torch.tensor(0.1))
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.8))

    def forward(self, x, cond, x_mask=None, cond_mask=None):
        """
        x:    (B, T_q, C) = query
        cond: (B, T_k, C) = key/value
        """
        text_metrics = {}

        # Pre-norm
        x_norm = self.pre_attn_norm(x)
        cond_norm = self.pre_cond_norm(cond)

        # Multi-head cross-attention
        attn_out, attn_weights = self.cross_attn(
            x_norm, cond_norm, cond_norm,
            key_padding_mask=cond_mask, need_weights=True,
            average_attn_weights=False  # ← add this
        )
        # attn_weights shape: (B, num_heads, T_q, T_k)

        # Possibly sharpen with temperature
        # (You could do something like: attn_weights = attn_weights.pow(1.0 / temp), renormalize, etc.)
        temp = self.attention_temperature.clamp(min=0.01, max=1.0)

        # For debugging, let’s gather stats about the attention:
        with torch.no_grad():
            b_sz, n_heads, t_q, t_k = attn_weights.shape
            attn_min = attn_weights.min().item()
            attn_max = attn_weights.max().item()
            attn_mean = attn_weights.mean().item()

            # The average (over heads) top-1 token each query attends to
            # We do it for the first example only for brevity
            if b_sz > 0:
                # shape = (num_heads, T_q)
                top_indices = torch.argmax(attn_weights[0], dim=-1)  # best token for each query step
                # e.g. top_indices is (n_heads, T_q)
            else:
                top_indices = None

        # Save to text_metrics
        text_metrics["attn_min"] = attn_min
        text_metrics["attn_max"] = attn_max
        text_metrics["attn_mean"] = attn_mean
        text_metrics["attn_temp"] = temp.item()
        if top_indices is not None:
            # to avoid spamming with large arrays, just show first head or slice
            text_metrics["top_indices_head0"] = top_indices[0, :5].tolist()  # show 5 for debug

        # Residual connection
        res_scale = self.residual_scale.clamp(min=0.5, max=1.0)
        x = x + res_scale * attn_out
        x = self.layer_norm(x)

        # FFN + residual
        ffn_out = self.ffn(x)
        x = x + res_scale * ffn_out
        x = self.layer_norm_ffn(x)

        return x, text_metrics


###############################################################################
# Encoder / Decoder
###############################################################################
class ConditionerEncoder(torch.nn.Module):
    """
    Takes the wave+mel features and downsamples them in multiple stages (ds_modules)
    to produce a 512-channel representation, passes it through a 2-layer bidirectional
    GRU, then (optionally) uses FiLM + cross-attention with text embeddings *after*
    the GRU block.
    """
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
        #
        # For text fusion
        film_global=None,
        mel_to_attn=None,
        cross_attention=None,
        attn_to_mel=None,
        text_direct_proj=None,
        text_impact_factor=None,
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

        # Additional stride conv to go from each stage's residual to final 512
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

        # final channel count, e.g. 512
        oc = input_channels * 2 ** len(ds_factors)

        self.seq_model = seq_model
        if seq_model == "gru":
            self.gru = torch.nn.GRU(
                oc,
                oc // 2,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
            )
            self.conv_block1 = ConvBlock(oc, act_type=act_type, use_weight_norm=use_weight_norm)
            self.conv_block2 = ConvBlock(oc, act_type=act_type, use_weight_norm=use_weight_norm)
        else:
            raise ValueError("Values for 'seq_model' can be only 'gru' in this setup")

        # Text-fusion modules
        self.film_global = film_global
        self.mel_to_attn = mel_to_attn
        self.cross_attention = cross_attention
        self.attn_to_mel = attn_to_mel
        self.text_direct_proj = text_direct_proj
        self.text_impact_factor = text_impact_factor

    def forward(
        self,
        x,
        x_mel,
        global_emb=None,
        seq_emb=None,
        train=False,
        current_epoch=0,
    ):
        """
        x:        (B, n_channels, T_wav)  # after input_conv
        x_mel:    (B, oc, T_mel)          # from MelAdapter, oc=512
        global_emb, seq_emb: from text encoder (may be None)
        """
        # Multi-scale downsampling
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

        # GRU block
        out, *_ = self.conv_block1(out)
        if self.with_gru_residual:
            res = out
        out_bt, *_ = self.gru(out.transpose(-2, -1))  # (B, T_mel, oc)
        out = out_bt.transpose(-2, -1)
        if self.with_gru_residual:
            out = (out + res) / math.sqrt(2)
        out, *_ = self.conv_block2(out)

        # text_metrics to track attention stats, etc.
        text_metrics = {}

        # Cross-attention & FiLM after GRU
        if (global_emb is not None or seq_emb is not None) and self.film_global is not None:
            # 1) FiLM with global embedding
            out_t = out.transpose(1, 2)  # (B, T, C)
            out_t, film_stats = self.film_global(out_t, global_emb)
            text_metrics.update(film_stats)

            # 2) Cross-attention with token-level embeddings
            if seq_emb is not None and self.cross_attention is not None:
                # Project to cross-attn dimension
                out_attn = self.mel_to_attn(out_t)

                # Each CrossAttentionBlock
                for i, layer in enumerate(self.cross_attention):
                    out_attn, layer_stats = layer(out_attn, seq_emb)
                    # prefix the metrics with block index
                    for k, v in layer_stats.items():
                        text_metrics[f"attn_block{i}_{k}"] = v

                # Back to original dimension
                out_t_ca = self.attn_to_mel(out_attn)

                # Combine FiLM and cross-attn result
                if train and current_epoch < 10:
                    film_w, ca_w = 0.7, 0.3
                elif train and current_epoch < 20:
                    film_w, ca_w = 0.6, 0.4
                else:
                    film_w, ca_w = 0.5, 0.5
                out_t_combined = film_w * out_t + ca_w * out_t_ca

                # Optionally add a small direct contribution from global text
                if (self.text_direct_proj is not None) and (current_epoch > 5):
                    direct_proj = self.text_direct_proj(global_emb)  # (B, C)
                    direct_proj = direct_proj.unsqueeze(-1).expand(
                        -1, -1, out_t_combined.shape[1]
                    )  # (B, C, T)
                    direct_scale = 0.1
                    out_t_combined = out_t_combined + direct_scale * direct_proj.transpose(1, 2)

                if self.text_impact_factor is not None:
                    raw_factor = torch.sigmoid(self.text_impact_factor)
                    blend_factor = 0.25 * raw_factor  # up to 0.25
                    out_t = (1.0 - blend_factor) * out_t + blend_factor * out_t_combined
                    text_metrics["text_impact_factor"] = raw_factor.item()
                    text_metrics["blend_factor"] = blend_factor.item()
                else:
                    out_t = out_t_combined

                text_metrics["film_weight"] = film_w
                text_metrics["ca_weight"] = ca_w

            out = out_t.transpose(1, 2)

        return out, lengths[::-1], text_metrics


class ConditionerDecoder(torch.nn.Module):
    """
    The upsampling path (mirrors ConditionerEncoder). Takes the
    final encoder output + multi-scale lengths to produce time-aligned
    feature maps, plus a list of 'conditions' at each scale.
    """
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
        """
        x: (B, 32*2^N, T//(product_of_rate_factors))
        lengths: reversed list of original time lengths at each stage
        """
        conditions = []
        x, *_ = self.input_conv_block(x)

        for up, length in zip(self.up_modules, lengths):
            x, _, cond = up(x, length=length)
            conditions.append(cond)
        return x, conditions


###############################################################################
# The top-level ConditionerNetwork
###############################################################################
class ConditionerNetwork(torch.nn.Module):
    """
    The main wrapper that:
      1) Takes a waveform x (and optional text),
      2) Produces a mel feature x_mel via MelAdapter,
      3) Runs down/GRU/up, possibly fusing text (inside the encoder),
      4) Outputs a multi-scale 'conditions' plus an optional signal y_hat.
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
        output_channels=None,
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        #
        # Text-fusion config
        text_encoder_config=None,
        film_global_dim=256,
        cross_attention_dim=256,
    ):
        super().__init__()

        print("\n========== ConditionerNetwork: Init ==========")
        print(f"Input Channels: {input_channels}, Output Channels: {output_channels}")
        print(f"n_channels: {n_channels}, rate_factors: {rate_factors}")
        print(f"n_mels={n_mels}, mel_oversample={n_mel_oversample}")
        print(f"seq_model={seq_model}, extra_conv_block={extra_conv_block}")
        print(f"Text encoder? {'YES' if text_encoder_config else 'NO'}")

        self.input_conv = cond_weight_norm(
            torch.nn.Conv1d(input_channels, n_channels, kernel_size=fb_kernel_size, padding="same"),
            use=use_weight_norm,
        )
        if output_channels is not None:
            self.output_conv = cond_weight_norm(
                torch.nn.Conv1d(
                    n_channels, output_channels, kernel_size=fb_kernel_size, padding="same"
                ),
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

        self.precoding = instantiate(precoding, _recursive_=True) if precoding else None

        # Optional text encoder
        self.text_encoder = None
        film_global = None
        mel_to_attn = None
        cross_attention = None
        attn_to_mel = None
        text_direct_proj = None
        text_impact_factor = None

        if text_encoder_config is not None:
            # Instantiate the text encoder
            self.text_encoder = instantiate(text_encoder_config, _recursive_=False)
            print("[DEBUG] Created text_encoder:", self.text_encoder)

            # Prepare FiLM + cross-attn modules for the Encoder
            film_global = FiLM(
                condition_dim=film_global_dim, feature_channels=total_channels
            )
            mel_to_attn = torch.nn.Linear(total_channels, cross_attention_dim)
            attn_to_mel = torch.nn.Linear(cross_attention_dim, total_channels)
            torch.nn.init.xavier_uniform_(mel_to_attn.weight)
            torch.nn.init.zeros_(mel_to_attn.bias)
            torch.nn.init.xavier_uniform_(attn_to_mel.weight)
            torch.nn.init.zeros_(attn_to_mel.bias)

            # A small stack of cross-attention layers
            cross_attention = torch.nn.ModuleList([
                CrossAttentionBlock(hidden_dim=cross_attention_dim, num_heads=8),
                CrossAttentionBlock(hidden_dim=cross_attention_dim, num_heads=8),
            ])

            text_direct_proj = torch.nn.Linear(film_global_dim, total_channels)
            torch.nn.init.xavier_uniform_(text_direct_proj.weight)
            torch.nn.init.zeros_(text_direct_proj.bias)

            # Learnable factor for controlling text impact
            text_impact_factor = torch.nn.Parameter(torch.tensor(0.4))

        # Build the Encoder/Decoder with text-fusion modules injected
        self.encoder = ConditionerEncoder(
            ds_factors=rate_factors,
            input_channels=n_channels,
            with_gru_residual=encoder_gru_residual,
            with_extra_conv_block=extra_conv_block,
            act_type=encoder_act_type,
            use_weight_norm=use_weight_norm,
            seq_model=seq_model,
            use_antialiasing=use_antialiasing,
            #
            film_global=film_global,
            mel_to_attn=mel_to_attn,
            cross_attention=cross_attention,
            attn_to_mel=attn_to_mel,
            text_direct_proj=text_direct_proj,
            text_impact_factor=text_impact_factor,
        )
        self.decoder = ConditionerDecoder(
            up_factors=rate_factors[::-1],
            input_channels=n_channels,
            with_extra_conv_block=extra_conv_block,
            act_type=decoder_act_type,
            use_weight_norm=use_weight_norm,
            use_antialiasing=use_antialiasing,
        )

        print("========== ConditionerNetwork: Done Init ==========\n")


    def forward(self, x, x_wav=None, train=False, text=None, current_epoch=0):
        """
        x:        (B, 1, T) or (B, input_channels, T)
        x_wav:    same shape as x if it’s transformed externally
        text:     a list of transcripts or None

        Returns: conditions, (optionally) y_hat, h, text_metrics, global_emb
        """
        text_metrics = {}

        n_samples = x.shape[-1]
        if x_wav is None:
            x_wav = x

        # 1) Compute mel features from the waveform
        x_mel = self.input_mel(x_wav)  # shape ~ (B, total_channels, T/160)

        # 2) Possibly run the text encoder
        global_emb = None
        seq_emb = None
        if self.text_encoder is not None and text is not None:
            # Put text encoder in correct mode
            self.text_encoder.train(mode=train)
            valid_text = [t for t in text if t and t.strip()]
            if len(valid_text) == 0:
                print("[DEBUG] No non-empty transcripts; skipping text encoder.")
            else:
                global_emb, seq_emb = self.text_encoder(valid_text)
                # Log some stats
                text_metrics["global_emb_min"] = global_emb.min().item()
                text_metrics["global_emb_max"] = global_emb.max().item()
                text_metrics["global_emb_mean"] = global_emb.mean().item()
                text_metrics["global_emb_std"] = global_emb.std().item()
                text_metrics["seq_emb_shape"] = list(seq_emb.shape)  # e.g. (B, T_tokens, c_dim)

        # 3) Optional precoding on x
        if self.precoding:
            x = self.precoding(x)

        # 4) Input conv to get (B, n_channels, T)
        x = self.input_conv(x)

        # 5) Encoder → merges with x_mel, runs GRU, does text fusion
        h, lengths, encoder_text_metrics = self.encoder(
            x, x_mel,
            global_emb=global_emb,
            seq_emb=seq_emb,
            train=train,
            current_epoch=current_epoch,
        )
        text_metrics.update(encoder_text_metrics)

        # 6) Decoder
        y_hat, conditions = self.decoder(h, lengths)

        # 7) Optional final conv
        if self.output_conv is not None:
            y_hat = self.output_conv(y_hat)

        # 8) Undo precoding if set
        if self.precoding:
            y_hat = self.precoding.inv(y_hat)

        # 9) Pad to original length
        y_hat = torch.nn.functional.pad(y_hat, (0, n_samples - y_hat.shape[-1]))

        if train:
            # Return (conditions, y_hat, h, plus any logging info)
            return conditions, y_hat, h, text_metrics, global_emb
        else:
            # Return a simplified set
            return conditions, text_metrics, global_emb
