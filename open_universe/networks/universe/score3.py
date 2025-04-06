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
The UNIVERSE score module

Author: Robin Scheibler (@fakufaku)
"""
import torch
from hydra.utils import instantiate

from .blocks import ConvBlock, PReLU_Conv, cond_weight_norm
from .sigma_block import SigmaBlock, SimpleTimeEmbedding


class ScoreEncoder(torch.nn.Module):
    def __init__(
        self,
        ds_factors,
        input_channels,
        noise_cond_dim,
        with_gru_conv_sandwich=False,
        with_extra_conv_block=False,
        act_type="prelu",
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
    ):
        super().__init__()

        c = input_channels
        self.extra_conv_block = with_extra_conv_block

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

        self.cond_proj = torch.nn.ModuleList(
            [
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, c * 2 ** (i + 1)),
                    use=use_weight_norm,
                )
                for i in range(len(ds_factors))
            ]
        )

        oc = input_channels * 2 ** len(ds_factors)  # num. channels bottleneck

        if self.extra_conv_block:
            self.ds_modules.append(
                ConvBlock(oc, act_type=act_type, use_weight_norm=use_weight_norm)
            )
            self.cond_proj.append(
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, 2 * oc),
                    use=use_weight_norm,
                )
            )

        self.seq_model = seq_model
        if seq_model == "gru":
            self.gru = torch.nn.GRU(
                oc,  # number of channels after downsampling
                oc // 2,  # bi-directional double # of output channels
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )

            self.gru_conv_sandwich = with_gru_conv_sandwich
            if self.gru_conv_sandwich:
                self.conv_block1 = ConvBlock(
                    oc, act_type=act_type, use_weight_norm=use_weight_norm
                )
                self.conv_block2 = ConvBlock(
                    oc, act_type=act_type, use_weight_norm=use_weight_norm
                )
        elif seq_model == "none":
            pass
        else:
            raise ValueError("Values for 'seq_model' can be gru|attention|none")

    def forward(self, x, noise_cond):
        residuals = []
        lengths = []
        for idx, (ds, lin) in enumerate(zip(self.ds_modules, self.cond_proj)):
            nc = lin(noise_cond)
            lengths.append(x.shape[-1])
            x, res, _ = ds(x, noise_cond=nc)
            residuals.append(res)

        if self.seq_model == "gru":
            if self.gru_conv_sandwich:
                x, *_ = self.conv_block1(x)
            x, _ = self.gru(x.transpose(-2, -1))
            x = x.transpose(-2, -1)
            if self.gru_conv_sandwich:
                x, *_ = self.conv_block2(x)
        elif self.seq_model == "attention":
            x = self.att(x)
        elif self.seq_model == "none":
            pass

        # return the residuals in reverse order to make it easy to use them in
        # the decoder
        return x, residuals[::-1], lengths[::-1]


class ScoreDecoder(torch.nn.Module):
    def __init__(
        self,
        up_factors,
        input_channels,
        noise_cond_dim,
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

        self.up_modules = torch.nn.ModuleList()
        self.noise_cond_proj = torch.nn.ModuleList()
        self.signal_cond_proj = torch.nn.ModuleList()

        if self.extra_conv_block:
            # adds extra input block with constant channels
            oc = input_channels * 2 ** len(up_factors)
            self.up_modules.append(
                ConvBlock(oc, act_type=act_type, use_weight_norm=use_weight_norm)
            )
            self.noise_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, 2 * oc),
                    use=use_weight_norm,
                )
            )
            self.signal_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Conv1d(oc, oc, kernel_size=1),
                    use=use_weight_norm,
                )
            )

        for c, r in zip(n_channels, up_factors):
            self.up_modules.append(
                ConvBlock(
                    c,
                    r,
                    "up",
                    act_type=act_type,
                    use_weight_norm=use_weight_norm,
                    antialiasing=use_antialiasing,
                )
            )
            self.noise_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Linear(noise_cond_dim, 2 * c),
                    use=use_weight_norm,
                )
            )
            self.signal_cond_proj.append(
                cond_weight_norm(
                    torch.nn.Conv1d(c, c, kernel_size=1),
                    use=use_weight_norm,
                )
            )

    def forward(self, x, noise_cond, input_cond, residuals, lengths):
        for lvl, (up, n_lin, s_lin, cond, res, length) in enumerate(
            zip(
                self.up_modules,
                self.noise_cond_proj,
                self.signal_cond_proj,
                input_cond,
                residuals,
                lengths,
            )
        ):
            nc = n_lin(noise_cond)
            sc = s_lin(cond)
            x, *_ = up(x, noise_cond=nc, input_cond=sc, res=res, length=length)
        return x


class TextAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        """
        Simple text attention layer for the score network bottleneck.
        
        Args:
            hidden_dim: Dimensionality of the features
            num_heads: Number of attention heads
        """
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.self_attn = torch.nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
    def forward(self, x, text_embedding):
        """
        Apply text-based attention to bottleneck features.
        
        Args:
            x: Encoder bottleneck features [B, C, T]
            text_embedding: Text embeddings [B, seq_len, embed_dim]
            
        Returns:
            Updated features with text attention
        """
        # Prepare for attention (convert to [B, T, C])
        x_t = x.transpose(1, 2)
        x_norm = self.layer_norm(x_t)
        
        # Apply cross-attention
        attn_out, _ = self.self_attn(
            query=x_norm,
            key=text_embedding,
            value=text_embedding
        )
        
        # Residual connection and back to [B, C, T]
        x_t = x_t + attn_out
        return x_t.transpose(1, 2)

class ScoreNetwork(torch.nn.Module):
    def __init__(
        self,
        fb_kernel_size=3,
        rate_factors=[2, 4, 4, 5],
        n_channels=32,
        n_rff=32,
        noise_cond_dim=512,
        encoder_gru_conv_sandwich=False,
        extra_conv_block=False,
        encoder_act_type="prelu",
        decoder_act_type="prelu",
        precoding=None,
        input_channels=1,
        output_channels=1,
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        time_embedding=None,
        text_embedding_dim=None,
        use_text_conditioning=False,
    ):
        super().__init__()
        
        # Initialize training tracking variables
        self.current_epoch = 0  # Will be updated by parent module

        if time_embedding == "simple":
            self.sigma_block = SimpleTimeEmbedding(n_dim=noise_cond_dim)
        else:
            self.sigma_block = SigmaBlock(n_rff, noise_cond_dim)

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.input_conv = torch.nn.Conv1d(
            input_channels, n_channels, kernel_size=fb_kernel_size, padding="same"
        )
        self.encoder = ScoreEncoder(
            ds_factors=rate_factors,
            input_channels=n_channels,
            noise_cond_dim=noise_cond_dim,
            with_gru_conv_sandwich=encoder_gru_conv_sandwich,
            with_extra_conv_block=extra_conv_block,
            act_type=encoder_act_type,
            use_weight_norm=use_weight_norm,
            seq_model=seq_model,
            use_antialiasing=use_antialiasing,
        )
        self.decoder = ScoreDecoder(
            up_factors=rate_factors[::-1],
            input_channels=n_channels,
            noise_cond_dim=noise_cond_dim,
            with_extra_conv_block=extra_conv_block,
            act_type=decoder_act_type,
            use_weight_norm=use_weight_norm,
            use_antialiasing=use_antialiasing,
        )
        self.prelu = torch.nn.PReLU()
        self.output_conv = PReLU_Conv(
            n_channels,
            output_channels,
            kernel_size=fb_kernel_size,
            padding="same",
            use_weight_norm=use_weight_norm,
        )

        self.precoding = instantiate(precoding, _recursive_=True) if precoding else None
        
        # Add text conditioning components if enabled
        self.use_text_conditioning = use_text_conditioning
        if use_text_conditioning and text_embedding_dim is not None:
            # Project text embeddings to noise conditioning dimension
            self.text_proj = torch.nn.Linear(text_embedding_dim, noise_cond_dim)
            # Add a text gating mechanism to control influence
            self.text_gate = torch.nn.Parameter(torch.tensor(0.2))
            # Text attention in bottleneck
            self.text_attention = TextAttentionBlock(noise_cond_dim)
            
            # Initialize weights
            torch.nn.init.xavier_uniform_(self.text_proj.weight)
            torch.nn.init.zeros_(self.text_proj.bias)

    def forward(self, x, sigma, cond, text_embedding=None):
        n_samples = x.shape[-1]

        if self.precoding:
            x = self.precoding(x)

        g = self.sigma_block(torch.log10(sigma))
        
        # Add text conditioning if available
        if self.use_text_conditioning and text_embedding is not None:
            # Store original noise embedding for magnitude comparison
            g_orig = g.clone()
            
            # Project text embedding
            text_proj = self.text_proj(text_embedding)
            # Try to get epoch info from global variables in universe_gan_NS3
            try:
                # Import the module that has our global variable
                from ..universe.universe_gan_NS3 import CURRENT_EPOCH, GLOBAL_STEP
                current_epoch = CURRENT_EPOCH
                global_step = GLOBAL_STEP
                print(f"[EPOCH CHECK] Score network - Got epoch from universe_gan_NS3: {current_epoch}")
            except (ImportError, AttributeError):
                # Fallback: Start with zero epoch and gradually increase text impact
                current_epoch = 0
                global_step = 0
                print(f"[EPOCH CHECK] Warning: Could not access epoch from universe_gan_NS3")
                
                # If we have global_step, estimate epoch from that
                global_step = getattr(self, 'global_step', 0)
                if global_step > 0:
                    # Assume ~1000 steps per epoch
                    current_epoch = int(global_step / 1000)
                    print(f"[EPOCH CHECK] Score network - Estimated epoch from steps: {current_epoch}")
            
            # Training-aware gating with gradual ramp-up
            training_progress = min(1.0, current_epoch / 20.0)  # Ramp up over 20 epochs
            training_curve = training_progress ** 3  # Cubic curve for slower start
            
            # Gate the text influence (trainable parameter)
            # But scale it down significantly at start of training
            base_text_gate = torch.sigmoid(self.text_gate)
            text_weight = base_text_gate * training_curve * 0.1  # Very small initial influence
            
            # Debug info about training stage
            print(f"[SCORE DEBUG] Training progress: {training_progress:.4f} (epoch {current_epoch}/20)")
            print(f"[SCORE DEBUG] Raw text gate: {base_text_gate.item():.4f}, Effective gate: {text_weight.item():.4f}")
            
            # Calculate metrics for logging
            text_proj_mag = text_proj.abs().mean().item()
            g_orig_mag = g_orig.abs().mean().item()
            
            # Combine with noise embedding
            g = g * (1 - text_weight) + text_proj * text_weight
            
            # Calculate impact metrics
            g_with_text_mag = g.abs().mean().item()
            
            # Print debug info
            print(f"[SCORE DEBUG] Text gate: {text_weight.item():.4f} (0=audio only, 1=text only)")
            print(f"[SCORE DEBUG] Noise embedding mag: {g_orig_mag:.4f}")
            print(f"[SCORE DEBUG] Text projection mag: {text_proj_mag:.4f}")
            print(f"[SCORE DEBUG] Combined embedding mag: {g_with_text_mag:.4f}")
            print(f"[SCORE DEBUG] Feature diff magnitude: {(g - g_orig).abs().mean().item():.4f}")
            
            # Create metrics dict to be logged later
            self.text_score_metrics = {
                "score_text_gate_raw": base_text_gate.item(),
                "score_text_gate_effective": text_weight.item(),
                "score_training_progress": training_progress,
                "score_training_curve": training_curve,
                "score_noise_emb_mag": g_orig_mag,
                "score_text_proj_mag": text_proj_mag,
                "score_combined_mag": g_with_text_mag,
                "score_feature_diff": (g - g_orig).abs().mean().item(),
                "score_text_impact_ratio": (g - g_orig).abs().mean().item() / g_orig_mag,
                "score_epoch": current_epoch
            }
        
        x = self.input_conv(x)
        h, residuals, lengths = self.encoder(x, noise_cond=g)
        
        # Apply text attention in bottleneck if available
        if self.use_text_conditioning and text_embedding is not None:
            # Store original bottleneck for comparison
            h_orig = h.clone()
            
            # Use current_epoch from earlier in the method if available
            if 'current_epoch' not in locals():
                # Try to get epoch info from global variables in universe_gan_NS3
                try:
                    from ..universe.universe_gan_NS3 import CURRENT_EPOCH
                    current_epoch = CURRENT_EPOCH
                    print(f"[EPOCH CHECK] Bottleneck - Got epoch from universe_gan_NS3: {current_epoch}")
                except (ImportError, AttributeError):
                    # Fallback to zero if we can't access the module
                    current_epoch = 0
                    print("[EPOCH CHECK] Bottleneck - Could not access epoch from universe_gan_NS3")
            
            # Use existing training_progress and training_curve if already calculated
            if 'training_progress' not in locals() or 'training_curve' not in locals():
                # Calculate training progression values
                training_progress = min(1.0, current_epoch / 20.0)
                training_curve = training_progress ** 3
                
            # Debug print with simpler format
            print(f"[EPOCH CHECK] In bottleneck attention - current_epoch: {current_epoch}, progress: {training_progress:.4f}")
            
            # Apply text attention with gradual blend
            h_attn = self.text_attention(h, text_embedding)
            
            # Early in training, use very little text attention
            # Use a cubic curve to start slow and accelerate
            attn_blend = training_curve
            h = h * (1 - attn_blend) + h_attn * attn_blend
            
            # Debug info
            print(f"[SCORE DEBUG] Bottleneck attention blend: {attn_blend:.4f} (higher with training)")
            
            # Calculate impact metrics
            h_orig_mag = h_orig.abs().mean().item()
            h_with_text_mag = h.abs().mean().item()
            h_diff_mag = (h - h_orig).abs().mean().item()
            h_impact_ratio = h_diff_mag / h_orig_mag
            
            # Debug prints
            print(f"[SCORE DEBUG] Bottleneck before attn mag: {h_orig_mag:.4f}")
            print(f"[SCORE DEBUG] Bottleneck after attn mag: {h_with_text_mag:.4f}")
            print(f"[SCORE DEBUG] Bottleneck diff magnitude: {h_diff_mag:.4f}")
            print(f"[SCORE DEBUG] Bottleneck impact ratio: {h_impact_ratio:.4f}")
            
            # Add to metrics dict
            if hasattr(self, 'text_score_metrics'):
                self.text_score_metrics.update({
                    "score_bottleneck_orig_mag": h_orig_mag,
                    "score_bottleneck_with_text_mag": h_with_text_mag,
                    "score_bottleneck_diff_mag": h_diff_mag,
                    "score_bottleneck_impact_ratio": h_impact_ratio,
                    "score_bottleneck_attn_blend": attn_blend
                })
            else:
                self.text_score_metrics = {
                    "score_bottleneck_orig_mag": h_orig_mag,
                    "score_bottleneck_with_text_mag": h_with_text_mag,
                    "score_bottleneck_diff_mag": h_diff_mag,
                    "score_bottleneck_impact_ratio": h_impact_ratio,
                    "score_bottleneck_attn_blend": attn_blend
                }
            
        s = self.decoder(
            h, noise_cond=g, input_cond=cond, residuals=residuals, lengths=lengths
        )
        s = self.output_conv(self.prelu(s))

        if self.precoding and hasattr(self.precoding, "inv"):
            s = self.precoding(s, inv=True)

        # adjust length and dimensions
        s = torch.nn.functional.pad(s, (0, n_samples - s.shape[-1]))

        return s
