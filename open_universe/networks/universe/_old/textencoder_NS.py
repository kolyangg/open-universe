import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, vocab=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.vocab = vocab  # Optional: a dict mapping tokens to indices

    def forward(self, text_input):
        # text_input: (B, L) of token indices
        x = self.embedding(text_input)  # (B, L, embed_dim)
        _, (hn, _) = self.lstm(x)        # hn: (num_layers, B, hidden_dim)
        hn = hn[-1]                     # (B, hidden_dim)
        return self.fc(hn)              # (B, hidden_dim)
