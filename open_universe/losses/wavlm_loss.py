# losses/wavlm_lmos.py
import torch, torchaudio
from torch import nn
from transformers import WavLMModel

# If training data is 24 kHz, to add self.resample = torchaudio.transforms.Resample(24_000,16_000) in __init__.

class WavLMLMOSLoss(nn.Module):
    """
    Implements  Eq.(2) in the screenshot (LMOS):
        100 · ‖φ(y) − φ(ŷ)‖₂²   +   ‖ |STFT(y)| − |STFT(ŷ)| ‖₁
    where φ is the convolutional front‑end of WavLM.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        stft_n_fft: int = 1024,
        stft_hop: int = 256,
        feat_weight: float = 100.0,
    ):
        super().__init__()

        # ---------- frozen WavLM conv feature extractor ----------
        wavlm = WavLMModel.from_pretrained(model_name)
        self.feat_extractor = wavlm.feature_extractor   # conv stack only
        self.feat_extractor.eval().requires_grad_(False)
        del wavlm  # save RAM
        # ---------------------------------------------------------

        self.register_buffer("window", torch.hann_window(stft_n_fft))
        self.stft_n_fft = stft_n_fft
        self.stft_hop   = stft_hop
        self.feat_w     = feat_weight

        # WavLM expects 16 kHz.  Cheap sinc‑resampler for other rates.
        self.resample = None

    @torch.no_grad()
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T]  or  [B, T]  — 16 kHz waveform on *any* device
        returns: [B, C, F]
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        if self.resample is not None:
            x = self.resample(x)
        return self.feat_extractor(x)            # stays on caller’s device

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y_hat, y : waveforms, shape [B,1,T] or [B,T]  (same sample‑rate)
        """
        # (1) WavLM‑Conv feature MSE
        phi_y     = self._phi(y)
        phi_y_hat = self._phi(y_hat)
        l_feat = torch.mean((phi_y - phi_y_hat) ** 2)

        # (2) magnitude‑STFT L1
        if y.dim() == 3:    # [B,1,T] → [B,T]
            y, y_hat = y.squeeze(1), y_hat.squeeze(1)
        stft = lambda z: torch.stft(
            z,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop,
            window=self.window.to(z.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        ).abs()

        mag_y, mag_y_hat = stft(y), stft(y_hat)
        l_mag = torch.mean(torch.abs(mag_y - mag_y_hat))

        return self.feat_w * l_feat + l_mag
