# test_attention.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import torchaudio

# Import the conditioner (assuming condition.py is in your Python path).
# Adjust the import to match your actual project structure:
# e.g. from open_universe.networks.universe.condition import ConditionerNetwork
from condition_NS_plbert5 import ConditionerNetwork


###############################################################################
# 1. Dummy Text Encoder (just returns random embeddings)
###############################################################################
class DummyTextEncoder(nn.Module):
    def __init__(self, global_dim=256, seq_dim=256, seq_len=6):
        """
        global_dim: dimension of "global" text embedding
        seq_dim: dimension of "per-token" embeddingsr
        seq_len: fixed number of tokens for demonstration
        """
        super().__init__()
        self.global_dim = global_dim
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        # Could define an actual learned embedding or a random-projection model
        # For demonstration, we just do nothing.

    def forward(self, text_batch):
        """
        text_batch: list of strings
        returns:
          global_emb:  (B, global_dim)
          seq_emb:     (B, seq_len, seq_dim)
        """
        batch_size = len(text_batch)
        # For real usage, you'd tokenize, then embed each text. Here, we just do random
        global_emb = torch.randn(batch_size, self.global_dim)
        seq_emb = torch.randn(batch_size, self.seq_len, self.seq_dim)
        return global_emb, seq_emb


###############################################################################
# 2. Tiny In-memory Dataset
###############################################################################


class TinySpeechTextDataset(torch.utils.data.Dataset):
    def __init__(self, audio_folder, txt_folder, sample_rate=24000):
        self.audio_folder = audio_folder
        self.txt_folder = txt_folder
        self.sample_rate = sample_rate
        self.audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith(".wav")])
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_filename = self.audio_files[idx]
        wav_path = os.path.join(self.audio_folder, audio_filename)
        txt_path = os.path.join(self.txt_folder, audio_filename.replace(".wav", ".txt"))
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        with open(txt_path, "r") as f:
            text = f.read().strip()
        return waveform, text

def collate_fn(batch):
    waves = [ex[0] for ex in batch]
    texts = [ex[1] for ex in batch]
    # In case waveforms have different lengths, pad them.
    waves = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
    return waves, texts





###############################################################################
# 3. Minimal Training Script
###############################################################################
def run_attention_debug():
    # -- 3.1 Create a small DataLoader
    
    audio_folder = "../../../../../data/vb_small/train/clean"   # <-- specify your audio folder
    txt_folder = "../../../../../data/vb_small/train/text"        # <-- specify your text folder

    dataset = TinySpeechTextDataset(audio_folder, txt_folder)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    # -- 3.2 Define a minimal text-encoder config and a ConditionerNetwork
    # We'll specify text_encoder_config as our DummyTextEncoder
    text_encoder_config = {
        "_target_": "open_universe.networks.universe.TextEncoder",  # or wherever it is located
        "global_dim": 64,  # smaller dims for speed
        "seq_dim": 64,
        "seq_len": 8,
        "hidden_dim": 256
    }

    # Minimal conditioner config:
    model = ConditionerNetwork(
        fb_kernel_size=3,
        rate_factors=[2, 4],  # just 2 downsampling stages for speed
        n_channels=16,
        n_mels=40,
        n_mel_oversample=4,
        encoder_gru_residual=False,
        extra_conv_block=False,
        encoder_act_type="prelu",
        decoder_act_type="prelu",
        precoding=None,      # no precoding
        input_channels=1,
        output_channels=1,   # produce 1-channel
        use_weight_norm=False,
        seq_model="gru",
        use_antialiasing=False,
        text_encoder_config=text_encoder_config,  # connect our dummy text encoder
        film_global_dim=64,
        cross_attention_dim=64,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -- 3.3 Define an optimizer and a dummy loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # We'll just run for a few epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch} ===")
        for step, (waves, texts) in enumerate(loader):
            waves = waves.to(device)
            # Our model expects x, x_wav, train=False/True, text=...
            # We'll set train=True to allow grad. current_epoch=epoch if you want
            conditions, y_hat, latent_h, text_metrics, g_emb = model(
                x=waves,
                x_wav=None,
                train=True,
                text=texts,
                current_epoch=epoch
            )

            # Just do a dummy reconstruction-style loss: compare y_hat to waves
            # because we have no real target. The point is to get backprop updates
            loss = loss_fn(y_hat, waves)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print losses and any attention stats
            print(f"step={step}, loss={loss.item():.5f}")
            # text_metrics has attention_min, attention_max, etc. if cross-attn is used
            # We'll just show them directly
            for k, v in text_metrics.items():
                if isinstance(v, float) or isinstance(v, int):
                    print(f"   {k} = {v:.4f}")
                elif isinstance(v, list):
                    print(f"   {k} = {v}")
            # (This helps confirm whether attention is forming peaked distributions)

    print("=== Done training. Check text_metrics prints above for attention logs! ===")


if __name__ == "__main__":
    run_attention_debug()
