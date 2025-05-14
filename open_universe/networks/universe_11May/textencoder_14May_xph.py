# textencoder_xphonebert.py
import re
import string
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

__all__ = ["TextEncoder"]


class TextEncoder(nn.Module):
    """
    Encode a batch of *phoneme strings* with XPhoneBERT.

    • input  : list[str]      (e.g. "P L IY1 Z K AO1 ...")
    • output : global_emb  – (B, hidden_dim)
              seq_emb     – (B, T, seq_dim)
              key_mask    – (B, T)  bool   True = ignore (pad / CLS / SEP)
    """

    MODEL_NAME = "vinai/xphonebert-base"

    def __init__(self,
                 hidden_dim: int,
                 seq_dim: int | None = None,
                 freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # ── backbone ────────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.backbone  = AutoModel.from_pretrained(self.MODEL_NAME)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── projection heads ───────────────────────────────────────────────────
        hsize = self.backbone.config.hidden_size
        if seq_dim is None:
            seq_dim = hidden_dim

        self.fc_global = nn.Linear(hsize, hidden_dim)
        self.fc_seq    = nn.Linear(hsize, seq_dim)

        self.global_norm = nn.LayerNorm(hsize)
        self.seq_norm    = nn.LayerNorm(hsize)

        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.xavier_uniform_(self.fc_seq.weight)
        nn.init.zeros_(self.fc_global.bias)
        nn.init.zeros_(self.fc_seq.bias)

        # ids for masking
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

    # ───────────────────────────────────────────────────────────────────────────
    # helpers
    # ───────────────────────────────────────────────────────────────────────────
    def _basic_clean(self, s: str) -> str:
        """Lower-case & remove punctuation (XPhoneBERT is case-insensitive)."""
        punc = f"[{re.escape(string.punctuation)}]"
        return re.sub(punc, "", s.lower()).strip()

    def decode_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    # ───────────────────────────────────────────────────────────────────────────
    # forward
    # ───────────────────────────────────────────────────────────────────────────
    def forward(self, sentences: List[str]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not sentences:
            raise ValueError("input list is empty")

        inputs = [self._basic_clean(s) for s in sentences]

        enc = self.tokenizer(
            inputs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        device         = next(self.parameters()).device
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # # debug - inputs, enc, input_ids, attention_mask
        # print("inputs :", inputs)
        # print("enc :", enc)
        # print("input_ids :", input_ids)
        # print("attention_mask :", attention_mask)
   

        # ------------------------------------------------------------- ❷
        # Run backbone under no-grad **only if it is frozen**
        # -------------------------------------------------------------
        if self.freeze_backbone:
            with torch.no_grad():
                hidden = self.backbone(
                    input_ids,
                    attention_mask=attention_mask
                ).last_hidden_state      # (B,T,H)
        else:
            hidden = self.backbone(
                input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

        cls_emb = hidden[:, 0, :]                       # (B,H)

        # projections (trainable, so keep grads) ------------------------------
        global_emb = self.fc_global(self.global_norm(cls_emb))
        seq_emb    = self.fc_seq(self.seq_norm(hidden))

        # mask CLS/SEP/PAD -----------------------------------------------------
        key_mask = (
            (input_ids == self.pad_id) |
            (input_ids == self.cls_id) |
            (input_ids == self.sep_id)
        )
        seq_emb = seq_emb.masked_fill(key_mask.unsqueeze(-1), 0.0)

        return global_emb, seq_emb, key_mask


# ─── simple test run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    enc = TextEncoder(hidden_dim=256).eval()
    sample = ["P L IY1 Z K AO1 L S T EH1 L AH0"]
    g, s, m = enc(sample)
    print("global :", g.shape, "seq :", s.shape, "mask :", m.shape)
