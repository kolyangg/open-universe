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

from transformers import WavLMModel, AutoFeatureExtractor 
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


# ----------------------------------------------------------------------
#  NEW WAVLM Adapter
#  (instead of MelAdapter)
# ----------------------------------------------------------------------

# class WavLMAdapter(torch.nn.Module):
#     def __init__(self, output_channels, ds_factor, oversample=2, use_weight_norm=False):
#         """
#         Parameters:
#           output_channels: the number of channels to output (e.g. 512)
#           ds_factor: the hop length used for the mel-based branch (e.g. 256)
#           oversample: multiplier for n_fft in MelAdapter (e.g. 2)
#         """
#         super().__init__()
#         self.ds_factor = ds_factor
#         self.oversample = oversample  # typically 2

#         # Initialize the feature extractor and model (assumes 16 kHz audio)
#         self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
#         self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")

#         # Convolution to map WavLM's 768 channels to the desired output_channels.
#         self.conv = cond_weight_norm(
#             torch.nn.Conv1d(768, output_channels, kernel_size=3, padding="same"),
#             use=use_weight_norm
#         )
#         self.conv_block = ConvBlock(output_channels, use_weight_norm=use_weight_norm)

#         # For consistency with the original MelAdapter:
#         n_fft = self.oversample * ds_factor  # e.g. 512 if ds_factor=256, oversample=2
#         pad_tot = n_fft - ds_factor           # e.g. 256
#         self.pad_left = pad_tot // 2           # e.g. 128
#         self.pad_right = pad_tot - self.pad_left  # e.g. 128
        
        
#         # DEBUG PRINT
#         self.debug_print = False # True

#     def compute_wavlm_features(self, x):
#         # x: [B, 1, T]
#         if self.debug_print:
#             print("[DEBUG] Input x shape:", x.shape)
#         B, C, T_orig = x.shape

#         # Compute extra pad so that length is a multiple of ds_factor
#         r = T_orig % self.ds_factor
#         extra_pad = self.ds_factor - r if r != 0 else 0

#         # Apply fixed padding (same as in MelAdapter)
#         x_padded = F.pad(x, (self.pad_left, extra_pad + self.pad_right))
#         if self.debug_print:
#             print("[DEBUG] x_padded shape:", x_padded.shape)  # Expected: [B, 1, T_eff]

#         # T_eff: effective length after padding.
#         T_eff = T_orig + self.pad_left + self.pad_right + extra_pad

#         # Compute expected number of frames as in MelAdapter:
#         n_fft = self.oversample * self.ds_factor
#         expected_frames = (T_eff - n_fft) // self.ds_factor + 1
#         if self.debug_print:
#             print("[DEBUG] T_orig:", T_orig, "T_eff:", T_eff, "expected_frames:", expected_frames)

#         # Remove channel dimension so that input becomes [B, T_eff]
#         x_in = x_padded.squeeze(1)
#         if self.debug_print:
#             print("[DEBUG] x_in shape (after squeeze):", x_in.shape)

#         # Process with the feature extractor.
#         inputs = self.processor(x_in, sampling_rate=16000, return_tensors="pt", padding=True)
#         values = inputs.input_values
#         if self.debug_print:
#             print("[DEBUG] values shape from processor:", values.shape)

#         # Explicitly squeeze the extra channel if values is [B, 1, T]
#         if values.dim() == 3 and values.size(1) == 1:
#             if self.debug_print:
#                 print("[DEBUG] Squeezing the extra channel dimension from values...")
#             values = values.squeeze(1)
#             if self.debug_print:
#                 print("[DEBUG] values shape after squeezing:", values.shape)
#         elif values.dim() == 4:
#             # If shape is [B, 1, X, T] squeeze dim=1
#             if self.debug_print:
#                 print("[DEBUG] Detected 4D tensor, squeezing dimension 1...")
#             values = values.squeeze(1)
#             if self.debug_print:
#                 print("[DEBUG] values shape after squeezing:", values.shape)
#         elif values.dim() == 3 and values.size(1) > 1:
#             if self.debug_print:
#                 print("[DEBUG] Averaging over channels in values (dim=1 has size:", values.size(1), ")...")
#             values = values.mean(dim=1)
#             if self.debug_print:
#                 print("[DEBUG] values shape after averaging:", values.shape)
            
#         if self.debug_print:
#             print("[DEBUG] Final values shape before wavlm:", values.shape)

#         # Move values to the same device as the wavlm model.
#         values = values.to(self.wavlm.device)

#         # Forward through WavLM.
#         with torch.no_grad():
#             hidden = self.wavlm(values).last_hidden_state  # Expected shape: [B, frames_w, 768]
#         if self.debug_print:
#             print("[DEBUG] WavLM output hidden shape:", hidden.shape)

#         # Transpose to [B, 768, frames_w]
#         hidden = hidden.transpose(1, 2)
#         if self.debug_print:
#             print("[DEBUG] Transposed hidden shape:", hidden.shape)

#         # Global normalization.
#         norm = (hidden ** 2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).sqrt()
#         hidden = hidden / norm.clamp(min=1e-5)

#         current_frames = hidden.shape[-1]
#         if self.debug_print:
#             print("[DEBUG] Current frames:", current_frames, "Expected frames:", expected_frames)
        
#         if current_frames != expected_frames:
#             hidden = F.interpolate(hidden, size=expected_frames, mode="linear", align_corners=False)
#             if self.debug_print:
#                 print("[DEBUG] Hidden shape after interpolation:", hidden.shape)

#         return hidden


#     def forward(self, x):
#         # x: [B, 1, T]
#         features = self.compute_wavlm_features(x)
#         # features is [B, 768, T_expected]
#         x_conv = self.conv(features)
#         # Pass through an extra convolution block (which preserves time dim if padded 'same')
#         x_out, *_ = self.conv_block(x_conv)
#         return x_out
    
#     def compute_mel_spec(self, x):
#         """
#         Backward-compatible method.
#         Mimics the interface of MelAdapter.compute_mel_spec() by returning features
#         whose global normalization ensures unit average energy per frame.
#         In this implementation, we simply call forward(x).
#         """
#         return self.forward(x)


# import torch
# import torch.nn.functional as F
# from transformers import AutoFeatureExtractor, WavLMModel

# # Assume cond_weight_norm and ConvBlock are defined as before.

# import torch
# import torch.nn.functional as F
# from transformers import AutoFeatureExtractor, WavLMModel

# Assume cond_weight_norm and ConvBlock are defined as in your code.

# class WavLMAdapter(torch.nn.Module):
#     def __init__(self, output_channels, ds_factor, oversample=2, use_weight_norm=False):
#         """
#         output_channels: Number of output channels (e.g., 512)
#         ds_factor: Hop length (e.g., 256)
#         oversample: Multiplier for n_fft (e.g., 2)
#         """
#         super().__init__()
#         self.ds_factor = ds_factor
#         self.oversample = oversample

#         # Initialize WavLM processor and model (expects 16 kHz audio)
#         self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
#         self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")

#         # Map WavLM's 768 channels to output_channels.
#         self.conv = cond_weight_norm(
#             torch.nn.Conv1d(768, output_channels, kernel_size=3, padding="same"),
#             use=use_weight_norm
#         )
#         self.conv_block = ConvBlock(output_channels, use_weight_norm=use_weight_norm)

#         # Fixed padding parameters (mimicking original MelAdapter)
#         n_fft = self.oversample * ds_factor
#         pad_tot = n_fft - ds_factor
#         self.pad_left = pad_tot // 2
#         self.pad_right = pad_tot - self.pad_left

#         self.debug_print = False # True  # Set to True to see debug outputs

#     def compute_wavlm_features(self, x):
#         # x: [B, 1, T]
#         if self.debug_print:
#             print("[DEBUG] Input x shape:", x.shape)
#         B, _, T_orig = x.shape

#         # Compute extra pad such that T_orig becomes a multiple of ds_factor.
#         r = T_orig % self.ds_factor
#         extra_pad = self.ds_factor - r if r != 0 else 0

#         # Fixed padding (as in MelAdapter)
#         x_padded = F.pad(x, (self.pad_left, extra_pad + self.pad_right))
#         if self.debug_print:
#             print("[DEBUG] x_padded shape:", x_padded.shape)  # Expected: [B, 1, T_eff]
#         T_eff = T_orig + self.pad_left + self.pad_right + extra_pad

#         # Expected frames as computed in MelAdapter.
#         n_fft = self.oversample * self.ds_factor
#         expected_frames = (T_eff - n_fft) // self.ds_factor + 1
#         if self.debug_print:
#             print("[DEBUG] T_orig:", T_orig, "T_eff:", T_eff, "expected_frames:", expected_frames)

#         # Remove channel dimension: [B, T_eff]
#         x_in = x_padded.squeeze(1)
#         if self.debug_print:
#             print("[DEBUG] x_in shape (after squeeze):", x_in.shape)

#         # Convert each sample to NumPy (detach to avoid grad issues).
#         x_list = [sample.detach().cpu().numpy() for sample in x_in]
#         inputs = self.processor(x_list, sampling_rate=16000, return_tensors="pt", padding=True)
#         values = inputs.input_values
#         if self.debug_print:
#             print("[DEBUG] values shape from processor:", values.shape)
#         # values now should be [B, T_eff] (batch preserved)

#         values = values.to(self.wavlm.device)
#         with torch.no_grad():
#             hidden = self.wavlm(values).last_hidden_state  # [B, frames_w, 768]
#         if self.debug_print:
#             print("[DEBUG] WavLM output hidden shape:", hidden.shape)

#         # Transpose to [B, 768, frames_w]
#         hidden = hidden.transpose(1, 2)
#         if self.debug_print:
#             print("[DEBUG] Transposed hidden shape:", hidden.shape)

#         # Global normalization.
#         norm = (hidden ** 2).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).sqrt()
#         hidden = hidden / norm.clamp(min=1e-5)

#         current_frames = hidden.shape[-1]
#         if self.debug_print:
#             print("[DEBUG] Current frames:", current_frames, "Expected frames:", expected_frames)
        
#         # Upsample if current frames do not match expected_frames.
#         if current_frames != expected_frames:
#             hidden = F.interpolate(hidden, size=expected_frames, mode="linear", align_corners=False)
#             if self.debug_print:
#                 print("[DEBUG] Hidden shape after interpolation:", hidden.shape)
#         return hidden

#     def forward(self, x):
#         # x: [B, 1, T]
#         features = self.compute_wavlm_features(x)  # [B, 768, T_expected]
#         x_conv = self.conv(features)               # [B, output_channels, T_expected]
#         x_out, *_ = self.conv_block(x_conv)         # [B, output_channels, T_expected]
#         return x_out

#     def compute_mel_spec(self, x):
#         # Backward-compatible interface.
#         return self.compute_wavlm_features(x)



class WavLMAdapter(torch.nn.Module):
    """
    Produces WavLM‑Conv (or 1‑st transformer‑layer) features and maps them
    to `output_channels` so the rest of the network stays unchanged.
    """

    def __init__(
        self,
        output_channels: int,
        ds_factor: int,
        oversample: int = 2,
        use_weight_norm: bool = False,
        sample_rate: int = 24_000,
        model_name: str = "microsoft/wavlm-base",
        take_layer: str = "conv",        # "conv"  (best in the paper) or "transformer1"
    ):
        super().__init__()
        self.ds_factor = ds_factor
        self.take_layer = take_layer
        self.wavlm_sr = 16_000           # WavLM was pre‑trained at 16 kHz

        # Resample on‑the‑fly if the project keeps 24 kHz wavs
        self.resample = (
            torchaudio.transforms.Resample(sample_rate, self.wavlm_sr)
            if sample_rate != self.wavlm_sr else None
        )

        # ---  load & freeze WavLM  -------------------------------------------------
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        in_dim = (
            self.wavlm.config.conv_dim[-1]              # 512  (conv encoder)
            if take_layer == "conv"
            else self.wavlm.config.hidden_size          # 768  (transformer hidden)
        )

        # ---  local projection + ConvBlock  ----------------------------------------
        self.conv = cond_weight_norm(
            torch.nn.Conv1d(in_dim, output_channels, kernel_size=3, padding="same"),
            use=use_weight_norm,
        )
        self.conv_block = ConvBlock(output_channels, use_weight_norm=use_weight_norm)

        # ---  padding identical to MelAdapter  -------------------------------------
        n_fft = oversample * ds_factor
        pad_tot = n_fft - ds_factor
        self.pad_left, self.pad_right = pad_tot // 2, pad_tot - pad_tot // 2

    # ------------------------------------------------------------------ helpers ---
    @torch.inference_mode()
    def _wavlm_feats(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav : [B, 1, T] – already padded to match MelAdapter logic
        Returns:
            feats : [B, C_in, F]   (C_in = 512 or 768)
        """
        if self.resample is not None:
            wav = self.resample(wav)                     # still [B, 1, T′]

        wav = wav.squeeze(1)                             # [B, T′]

        if self.take_layer == "conv":
            feats = self.wavlm.feature_extractor(wav)    # [B, C_in, F]
        else:  # first transformer layer (index 1; 0 is CNN‑proj)
            h = self.wavlm(
                wav, output_hidden_states=True, return_dict=True
            ).hidden_states[1]                           # [B, F, C_in]
            feats = h.transpose(1, 2)                    # [B, C_in, F]

        return feats

    # -----------------------------------------------------------------  forward ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 1, T]  –  24 kHz waveform (same as before)
        Returns:
            out : [B, output_channels, ⌈T/ds_factor⌉]
        """
        # replicate MelAdapter padding so frame count matches
        r = x.shape[-1] % self.ds_factor
        extra = self.ds_factor - r if r else 0
        x = F.pad(x, (self.pad_left, extra + self.pad_right))

        feats = self._wavlm_feats(x)                     # [B, C_in, F_wavlm]

        # interpolate to the exact #mel‑frames: F_target = ⌈T/ds_factor⌉
        target_len = math.ceil(x.shape[-1] / self.ds_factor)
        feats = F.interpolate(feats, size=target_len, mode="nearest")

        # same simple global‑energy normalisation used in MelAdapter
        norm = (feats.pow(2).sum(dim=-2, keepdim=True)
                          .mean(dim=-1, keepdim=True).sqrt())
        feats = feats / norm.clamp(min=1e-5)

        # local processing (unchanged downstream interface)
        feats = self.conv(feats)
        feats, *_ = self.conv_block(feats)
        return feats


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
        # self.input_mel = MelAdapter(
        #     n_mels,
        #     total_channels,
        #     total_ds * input_channels,
        #     n_mel_oversample,
        #     use_weight_norm=use_weight_norm,
        # )
        
        self.input_mel = WavLMAdapter(
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

    def forward(self, x, x_wav=None, train=False, text = None):
        n_samples = x.shape[-1]

        if x_wav is None:
            # this is used in case some type of transform is appled to
            # x before input.
            # This way, we can pass the original waveform
            x_wav = x

        x_mel = self.input_mel(x_wav)

        if self.precoding:
            x = self.precoding(x)  # do this after mel-spec comp

        x = self.input_conv(x)
        h, lengths = self.encoder(x, x_mel)  # latent representation

        y_hat, conditions = self.decoder(h, lengths)

        if self.output_conv is not None:
            y_hat = self.output_conv(y_hat)

        if self.precoding:
            y_hat = self.precoding.inv(y_hat)

        # adjust length and dimensions
        y_hat = torch.nn.functional.pad(y_hat, (0, n_samples - y_hat.shape[-1]))

        if train:
            return conditions, y_hat, h
        else:
            return conditions
