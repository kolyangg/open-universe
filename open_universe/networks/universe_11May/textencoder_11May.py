import os
import sys
import re
import string
import warnings
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer

# ─── Allow the custom classes stored in the PL‑BERT checkpoint ────────────────
from torch.serialization import add_safe_globals
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer
add_safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer])
# ───────────────────────────────────────────────────────────────────────────────

# Allow remote code if needed
os.environ["TRUST_REMOTE_CODE"] = "True"

# Define the PLBERT path relative to this file
PLBERT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../../_miipher/miipher2.0/plbert/")
)
if PLBERT_PATH not in sys.path:
    sys.path.insert(0, PLBERT_PATH)

# Instead of using EspeakBackend, we use OpenPhonemizer
from openphonemizer import OpenPhonemizer
from text_utils import TextCleaner


class TextEncoder(nn.Module):
    """Encode a batch of sentences to global & per‑token embeddings using PL‑BERT.

    The debug utilities now use the **same symbol table** as ``TextCleaner`` so the
    decoded tokens reflect the actual IPA/BPE symbols rather than unrelated words.
    """

    def __init__(self, hidden_dim: int, seq_dim: int | None = None, freeze_plbert: bool = True):
        super().__init__()

        # ── PL‑BERT -----------------------------------------------------------------
        plbert_root = PLBERT_PATH
        log_dir = os.path.join(plbert_root, "Checkpoint")
        config_path = os.path.join(log_dir, "config.yml")
        plbert_config = yaml.safe_load(open(config_path, "r"))
        albert_config = AlbertConfig(**plbert_config["model_params"])
        self.plbert = AlbertModel(albert_config)

        # latest checkpoint
        ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("step_")]
        iters = sorted(int(f.split("_")[-1].split(".")[0]) for f in ckpt_files)[-1]
        checkpoint_path = os.path.join(log_dir, f"step_{iters}.t7")
        print("Loading PL-BERT checkpoint from:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # strip "module.encoder." from keys
        state_dict = checkpoint["net"]
        new_state_dict = {
            k[15:]: v  # len("module.encoder.") == 15
            for k, v in state_dict.items()
            if k.startswith("module.encoder.") and k[15:] in self.plbert.state_dict()
        }
        self.plbert.load_state_dict(new_state_dict, strict=False)

        # ── Tokenizers --------------------------------------------------------------
        # * TransfoXLTokenizer is still kept for completeness but only used as
        #   a fallback in decode_tokens().
        self.tokenizer = TransfoXLTokenizer.from_pretrained(
            plbert_config["dataset_params"]["tokenizer"]
        )

        # Phonemizer + cleaner -------------------------------------------------------
        self.text_cleaner = TextCleaner()
        self.phonemizer = OpenPhonemizer()

        # keep the id of the plain-space symbol for masking later
        self.space_id = self.text_cleaner.word_index_dictionary[" "]

        # Build reverse symbol lookup for **debug prints only** ---------------------
        self.id2symbol: dict[int, str] | None = {
            idx: ch for ch, idx in self.text_cleaner.word_index_dictionary.items()
        }
        if not self.id2symbol:
            warnings.warn(
                "Phoneme vocabulary not found – debug prints will be wrong",
                RuntimeWarning,
            )

        # ── Projection heads --------------------------------------------------------
        self.fc_global = nn.Linear(self.plbert.config.hidden_size, hidden_dim)
        if seq_dim is None:
            seq_dim = hidden_dim
        self.fc_seq = nn.Linear(self.plbert.config.hidden_size, seq_dim)

        # Layer norms improve stability (added 08 May) -----------------------------
        self.seq_norm = nn.LayerNorm(self.plbert.config.hidden_size)
        self.global_norm = nn.LayerNorm(self.plbert.config.hidden_size)

        # Xavier init on new layers -------------------------------------------------
        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.xavier_uniform_(self.fc_seq.weight)
        nn.init.zeros_(self.fc_global.bias)
        nn.init.zeros_(self.fc_seq.bias)

        # Optionally freeze PL‑BERT --------------------------------------------------
        if freeze_plbert:
            for p in self.plbert.parameters():
                p.requires_grad = False

        # phoneme‑to‑id cache --------------------------------------------------------
        self.phoneme_cache: dict[str, str] = {}

    # ───────────────────────────────────────────────────────────────────────────────
    # Debug helpers
    # ───────────────────────────────────────────────────────────────────────────────
    def decode_tokens(self, token_ids: list[int]) -> list[str]:
        """Convert integer IDs back to their symbol strings for logging only."""
        if self.id2symbol is not None:
            return [self.id2symbol.get(i, f"<UNK:{i}>") for i in token_ids]
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return self.tokenizer.convert_ids_to_tokens(token_ids)
        return [f"<ID:{i}>" for i in token_ids]

    # ───────────────────────────────────────────────────────────────────────────────
    # Pre‑processing
    # ───────────────────────────────────────────────────────────────────────────────
    def tokenize(self, sents: list[str]):
        batched: list[torch.LongTensor] = []
        max_len = 0

        print(f"[DEBUG] Original sentences: {sents}")

        for sent in sents:
            # Empty sentence guard --------------------------------------------------
            if not sent or not sent.strip():
                ids = torch.LongTensor([0])  # placeholder pad token
                batched.append(ids)
                max_len = max(max_len, 1)
                print("[DEBUG] Empty string detected, using placeholder token ID: [0]")
                continue

            # Phonemize (cached) ----------------------------------------------------
            phonemes = self.phoneme_cache.get(sent)
            if phonemes is None:
                try:
                    phonemes = self.phonemizer(sent)
                except Exception as e:  # fall back to raw text
                    warnings.warn(f"Phonemizer failed on '{sent}': {e}")
                    phonemes = sent
                self.phoneme_cache[sent] = phonemes

            print(f"[DEBUG] IPA  : '{phonemes}'")

            # Clean & map to IDs ----------------------------------------------------
            # ids_list = self.text_cleaner(phonemes)


            ids_raw = self.text_cleaner(phonemes)
            # ── keep *only the first* occurrence of every symbol ──────────────
            seen = set()
            ids_list = []
            for t in ids_raw:
                if t not in seen:
                    ids_list.append(t)
                    seen.add(t)
            if len(ids_raw) != len(ids_list):
                print(f"[DEBUG] uniq-filter: {len(ids_raw)}→{len(ids_list)} tokens")


            ids = torch.LongTensor(ids_list)

            # Debug prints ---------------------------------------------------------
            print_limit = 20
            print(f"[DEBUG] IDs  : {ids_list[:print_limit]}... (len={len(ids_list)})")
            print(f"[DEBUG] Syms : {self.decode_tokens(ids_list[:print_limit])}")

            batched.append(ids)
            max_len = max(max_len, len(ids))

        # Pad into a tensor ---------------------------------------------------------
        B = len(sents)
        phoneme_ids = torch.zeros((B, max_len), dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.float)
        for i, ids in enumerate(batched):
            phoneme_ids[i, : len(ids)] = ids
            mask[i, : len(ids)] = 1.0

        print(f"[DEBUG] Final phoneme_ids shape: {phoneme_ids.shape}")
        return phoneme_ids, mask

    # ───────────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ───────────────────────────────────────────────────────────────────────────────
    def forward(self, input_data: list[str]):
        """input_data is a list of raw sentences."""
        # quick text clean: lower + strip + remove punctuation ----------------------
        punct_re = re.compile(f"[{re.escape(string.punctuation)}]")
        input_data = [punct_re.sub("", s.lower()).strip() for s in input_data]
        print(f"[DEBUG] Preprocessed input: {input_data[:2]}…")

        phoneme_ids, attention_mask = self.tokenize(input_data)

        # device move --------------------------------------------------------------
        device = next(self.plbert.parameters()).device
        phoneme_ids = phoneme_ids.to(device)
        attention_mask = attention_mask.to(device)

        # PL‑BERT ------------------------------------------------------------------
        outputs = self.plbert(phoneme_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        global_emb = self.fc_global(self.global_norm(cls_emb))
        seq_emb = self.fc_seq(self.seq_norm(outputs.last_hidden_state))
        

        # key_mask = attention_mask == 0  # True → pad

        # Build a boolean mask for cross-attention
        #   – pad       -> True  (already 1 in attention_mask)
        #   – space id  -> True  (new)
        
        # build key-mask  (pad OR space)  →  True == “ignore”
        key_mask = (attention_mask == 0) | (phoneme_ids == self.space_id)
        # zero-out values of ignored tokens so they cannot dominate attention
        seq_emb = seq_emb.masked_fill(key_mask.unsqueeze(-1), 0.0)



        # stats --------------------------------------------------------------------
        print(
            f"[DEBUG] Global emb stats: min={global_emb.min():.3f} max={global_emb.max():.3f} "
            f"mean={global_emb.mean():.3f}"
        )

        return global_emb, seq_emb, key_mask
