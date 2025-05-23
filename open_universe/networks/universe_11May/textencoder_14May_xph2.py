# textencoder_14May_xph.py
import re
import string
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from text2phonemesequence import Text2PhonemeSequence   # NEW

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
                 freeze_backbone: bool = True,
                 language: str = "eng-us"):
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
        
        
        
        # # # ── phoneme converter ────────────────────────────────────────────────
        # # Runs on GPU automatically if backbone is moved there
        # self.t2p = Text2PhonemeSequence(
        #     language=language,
        #     is_cuda=torch.cuda.is_available()
        # )   
        
        # # ── phoneme converter ────────────────────────────────────────────────
        # # Runs on GPU automatically if backbone is moved there
        # self.t2p = Text2PhonemeSequence(
        #     language=language,
        #     # is_cuda=torch.cuda.is_available(),
        #     is_cuda=False # easier to keep it on CPU
        # )   
        
        # ── phoneme converter (lazy - one per device) ───────────────────────
        self.language   = language
        self._t2p_cache: dict[str, Text2PhonemeSequence] = {}
        
    
    # --------------------------------------------------------------------- #
    # Build / fetch Text2PhonemeSequence that lives on *device*             #
    # --------------------------------------------------------------------- #
    def _get_t2p_on(self, device: torch.device) -> "Text2PhonemeSequence":
        key = str(device)
        # if key not in self._t2p_cache:
        #     use_cuda = device.type == "cuda"
        #     t2p = Text2PhonemeSequence(language=self.language, is_cuda=use_cuda)
        #     if use_cuda:
        #         t2p = t2p.to(device)
        #     self._t2p_cache[key] = t2p
        
        if key not in self._t2p_cache:
            if device.type == "cuda":
                # Build the phonemiser *while* the correct GPU is current.
                # Text2PhonemeSequence has no .to(), so we create one per card.
                with torch.cuda.device(device):
                    t2p = Text2PhonemeSequence(
                        language=self.language,
                        is_cuda=True,
                    )
            else:  # CPU
                t2p = Text2PhonemeSequence(
                    language=self.language,
                    is_cuda=False,
                )
            self._t2p_cache[key] = t2p
                
        
        return self._t2p_cache[key]


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

        # inputs = [self._basic_clean(s) for s in sentences]
        
        # Convert raw text → phoneme sequence (space-separated)
        # inputs = [self.t2p.infer_sentence(self._basic_clean(s))
        #           for s in sentences]                      # NEW / REPLACES OLD
    
        device = next(self.parameters()).device
        t2p    = self._get_t2p_on(device)

        inputs = []
        for s in sentences:
            cleaned = self._basic_clean(s)
            if cleaned == "":               # silence placeholder
                inputs.append("▁")
            else:
                inputs.append(t2p.infer_sentence(cleaned))


        enc = self.tokenizer(
            inputs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # device         = next(self.parameters()).device
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


# ──────────────────────────────────────────────────────────────────────────────
# debug / smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        "This is a test sentence .",
        "Another example to verify the pipeline !",
    ]

    encoder = TextEncoder(hidden_dim=256, freeze_backbone=True).eval()

    with torch.no_grad():
        for sent in sample:
            cleaned   = encoder._basic_clean(sent)
            phonemes  = encoder.t2p.infer_sentence(cleaned)
            ids       = encoder.tokenizer(phonemes,
                                          add_special_tokens=True)["input_ids"]
            tokens    = encoder.tokenizer.convert_ids_to_tokens(ids)

            print("-" * 60)
            print("SOURCE   :", sent)
            print("PHONEMES :", phonemes)
            print("IDS      :", ids)
            print("TOKENS   :", tokens)

        # run a forward pass to ensure nothing breaks
        encoder(sample)
        print("-" * 60)
        print("Forward pass OK ✔")