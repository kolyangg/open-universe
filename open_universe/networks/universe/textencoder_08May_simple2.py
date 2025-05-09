import os
import sys
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel

import os, re, string, pickle
from collections import Counter
from typing import List

# # ─── local imports ─────────────────────────────────────────────────────────────
# from simple_tokenizer import SimpleTokenizer   # ← your tokenizer file
# # ───────────────────────────────────────────────────────────────────────────────

# Allow remote code if needed
os.environ["TRUST_REMOTE_CODE"] = "True"

# Path to the (original) PL-BERT repo
PLBERT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '../../../../../_miipher/miipher2.0/plbert/')
)
if PLBERT_PATH not in sys.path:
    sys.path.insert(0, PLBERT_PATH)

from pathlib import Path
TOK_PATH = Path(__file__).parent / "../../../simple_tokenizer.pkl"   # same folder as script


import os
import re
import string
import pickle
from collections import Counter


class SimpleTokenizer:
    """
    Word-level tokenizer with:
      • special tokens  : <PAD> =0, <UNK>=1
      • optional fixed-size *word* vocab
      • guaranteed 1-char tokens <C_a> … <C_z>, <C_0> … <C_9>
        so an OOV word is split into characters instead of <UNK>.
    """

    CHAR_TOKENS = list(string.ascii_lowercase) + list(string.digits)

    def __init__(self, vocab_size: int | None = 10_000, add_char_fallback: bool = True):
        """
        Parameters
        ----------
        vocab_size : int | None
            Max *word* tokens *excluding* special + char tokens.
            None ⇒ unlimited.
        add_char_fallback : bool
            If True, reserves tokens <C_a> … and uses them when a word is OOV.
        """
        self.vocab_size = vocab_size
        self.add_char_fallback = add_char_fallback

        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.next_id = 2  # start after PAD, UNK

        # will be filled in fit()
        self.char_token_ids: dict[str, int] = {}

    # ────────────────────────────────────────────────────────────────────────
    # public API
    # ────────────────────────────────────────────────────────────────────────
    def fit(self, texts: List[str]):
        """Build vocabulary from a list of raw texts."""
        # 1) optional char tokens
        if self.add_char_fallback:
            for ch in self.CHAR_TOKENS:
                self._add_token(f"<C_{ch}>", force=True)   # always add

        # 2) word statistics
        counts = Counter()
        for txt in texts:
            counts.update(txt.lower().split())

        # 3) take most common words up to vocab_size
        limit = (self.vocab_size or len(counts))           # None → unlimited
        for word, _ in counts.most_common(limit):
            self._add_token(word)

        # store quick look-up for char fallback
        self.char_token_ids = {
            ch: self.word_to_id.get(f"<C_{ch}>") for ch in self.CHAR_TOKENS
        }

        print(f"Vocabulary built: {len(self.word_to_id)} tokens "
              f"(words ≤ {self.vocab_size}, chars × {len(self.CHAR_TOKENS)})")
        return self

    def tokenize(self, text: str) -> List[int]:
        """Convert a sentence to a list of token IDs with char fallback."""
        ids: List[int] = []
        for word in text.lower().split():
            tid = self.word_to_id.get(word)
            if tid is not None:
                ids.append(tid)
            elif self.add_char_fallback:
                # decompose into characters
                ids.extend(self.char_token_ids.get(ch, 1)  # 1 = <UNK> char
                           for ch in word)
            else:
                ids.append(1)      # <UNK>
        return ids or [1]           # never return []

    def decode(self, ids: List[int]) -> List[str]:
        """IDs → tokens (words or <C_x>)."""
        return [self.id_to_word.get(i, "<UNK>") for i in ids]

    # alias for HuggingFace-style API
    convert_ids_to_tokens = decode

    # ───────────────────────────────────────────── storage helpers ──────────
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls(vocab_size=state["vocab_size"],
                  add_char_fallback=state["add_char_fallback"])
        tok.__dict__.update(state)
        print(f"Tokenizer loaded from {path} ({len(tok.word_to_id)} tokens)")
        return tok

    # ───────────────────────────────────────────── private  ────────────────
    def _add_token(self, token: str, force: bool = False):
        """Add a single token to vocab (internal)."""
        if token in self.word_to_id:
            return
        if (self.vocab_size is not None
                and not force
                and (self.next_id - 2 - len(self.CHAR_TOKENS)) >= self.vocab_size):
            return      # reached word budget
        self.word_to_id[token] = self.next_id
        self.id_to_word[self.next_id] = token
        self.next_id += 1


# def main():
#     # Configuration
#     folder_path = "../../data/voicebank_demand/trainset_28spk_txt"  # Change this to your folder path
#     vocab_size = 1000
#     output_path = "simple_tokenizer.pkl"
    
#     # Process text files
#     texts = process_text_files(folder_path)
    
#     # Train tokenizer
#     tokenizer = SimpleTokenizer(vocab_size=vocab_size)
#     tokenizer.fit(texts)
    
#     # Save tokenizer
#     tokenizer.save(output_path)
    
#     # Example usage
#     if texts:
#         sample_text = texts[0][:100]  # First 100 chars of first text
#         print(f"\nSample text: '{sample_text}'")
        
#         tokens = tokenizer.tokenize(sample_text)
#         print(f"Tokenized: {tokens}")
        
#         decoded = tokenizer.decode(tokens)
#         print(f"Decoded: {' '.join(decoded)}")
        
#         # Vocabulary stats
#         print(f"\nVocabulary size: {len(tokenizer.word_to_id)}")
#         print(f"Top 10 words: {list(tokenizer.word_to_id.keys())[:12]}")  # First 12 includes <PAD> and <UNK>





class TextEncoder(nn.Module):
    """
    TextEncoder that feeds a *word-level* SimpleTokenizer directly into PL-BERT.

    Debug prints show:
      • pre-processed sentences
      • token-ID samples
      • decoded words
      • tensor shapes / padding stats
    """

    def __init__(
        self,
        hidden_dim: int,
        seq_dim: int | None = None,
        tokenizer_path: str = TOK_PATH,
        freeze_plbert: bool = True,
    ):
        super().__init__()

        # ─── Load PL-BERT backbone ────────────────────────────────────────────
        plbert_root = PLBERT_PATH
        log_dir = os.path.join(plbert_root, "Checkpoint")
        config_path = os.path.join(log_dir, "config.yml")
        plbert_config = yaml.safe_load(open(config_path, "r"))
        albert_config = AlbertConfig(**plbert_config["model_params"])
        self.plbert = AlbertModel(albert_config)

        # most-recent checkpoint
        ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("step_")]
        iters = sorted(int(f.split("_")[-1].split(".")[0]) for f in ckpt_files)[-1]
        ckpt_path = os.path.join(log_dir, f"step_{iters}.t7")
        print(f"[INFO] Loading PL-BERT checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # strip 'module.encoder.' prefix
        new_state = {
            k.replace("module.encoder.", ""): v
            for k, v in ckpt["net"].items()
            if k.startswith("module.encoder.")
        }
        self.plbert.load_state_dict(new_state, strict=False)

        # ─── Simple tokenizer ─────────────────────────────────────────────────
        self.tokenizer = SimpleTokenizer.load(tokenizer_path)
        # self.plbert.resize_token_embeddings(self.tokenizer.vocab_size)
        
        actual_vocab = len(self.tokenizer.word_to_id)   # total tokens incl. chars
        self.plbert.resize_token_embeddings(actual_vocab)
        print(f"[INFO] PL-BERT embedding resized to {actual_vocab}")

        # print(f"[INFO] SimpleTokenizer vocab size = {self.tokenizer.vocab_size}")

        # ─── Projection heads ─────────────────────────────────────────────────
        if seq_dim is None:
            seq_dim = hidden_dim

        self.fc_global = nn.Linear(self.plbert.config.hidden_size, hidden_dim)
        self.fc_seq   = nn.Linear(self.plbert.config.hidden_size, seq_dim)

        self.global_norm = nn.LayerNorm(self.plbert.config.hidden_size)
        self.seq_norm    = nn.LayerNorm(self.plbert.config.hidden_size)

        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.xavier_uniform_(self.fc_seq.weight)
        nn.init.zeros_(self.fc_global.bias)
        nn.init.zeros_(self.fc_seq.bias)

        if freeze_plbert:
            for p in self.plbert.parameters():
                p.requires_grad = False

    # ─────────────────────────────────────────────────────────────────────────
    # tokenization helpers
    # ─────────────────────────────────────────────────────────────────────────
    def decode_tokens(self, ids):
        """Return list[str] for debug printing"""
        return self.tokenizer.decode(ids)

    def tokenize(self, sentences):
        """Word-level tokenization + padding mask"""
        batched, max_len = [], 0

        for sent in sentences:
            if not sent or not sent.strip():          # empty → <UNK>
                ids = torch.tensor([1], dtype=torch.long)
            else:
                ids = torch.tensor(self.tokenizer.tokenize(sent),
                                    dtype=torch.long)
            batched.append(ids)
            max_len = max(max_len, len(ids))

        ids_tensor  = torch.zeros(len(batched), max_len, dtype=torch.long)
        mask_tensor = torch.zeros_like(ids_tensor, dtype=torch.float)

        for i, ids in enumerate(batched):
            ids_tensor[i, : len(ids)]  = ids
            mask_tensor[i, : len(ids)] = 1.0

        return ids_tensor, mask_tensor

    # ─────────────────────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode(False)
    def forward(self, texts: list[str]):
        """
        Args
        ----
        texts : list[str]
            Raw sentences.
        Returns
        -------
        global_emb : (B, hidden_dim)
        seq_emb    : (B, seq_len, seq_dim)
        text_key_mask : (B, seq_len)  True = padding
        """
        # basic lower-casing & punctuation cleanup (same as tokenizer.fit)
        punct_re = re.compile(rf"[{re.escape(string.punctuation)}]")
        cleaned = [" ".join(punct_re.sub(" ", t.lower()).split()) for t in texts]

        print(f"[DEBUG] Input sentences (first 2): {cleaned[:2]}")

        ids, attn_mask = self.tokenize(cleaned)

        # debug prints
        sample = ids[0][: min(10, ids.size(1))].tolist()
        print(f"[DEBUG] Token IDs sample: {sample}")
        print(f"[DEBUG] Decoded sample   : {self.decode_tokens(sample)}")
        print(f"[DEBUG] ids.shape = {tuple(ids.shape)}  "
              f"padding = {(1 - attn_mask.mean()).item()*100:.2f}%")

        # move to same device as pl-bert
        device = next(self.plbert.parameters()).device
        ids, attn_mask = ids.to(device), attn_mask.to(device)

        outputs = self.plbert(ids, attention_mask=attn_mask)

        cls_emb = outputs.last_hidden_state[:, 0]          # (B, H)
        global_emb = self.fc_global(self.global_norm(cls_emb))
        seq_emb    = self.fc_seq(self.seq_norm(outputs.last_hidden_state))

        text_key_mask = (attn_mask == 0)                   # invert mask

        return global_emb, seq_emb, text_key_mask
