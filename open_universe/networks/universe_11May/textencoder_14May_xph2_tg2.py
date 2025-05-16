# textencoder_14May_xph.py
import re
import string
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from text2phonemesequence import Text2PhonemeSequence   # NEW
import tgt # NEW

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
        
        
        
        # ── phoneme converter ────────────────────────────────────────────────
        # # Runs on GPU automatically if backbone is moved there
        # self.t2p = Text2PhonemeSequence(
        #     language=language,
        #     # is_cuda=torch.cuda.is_available(),
        #     is_cuda=False # easier to keep it on CPU
        # )   
        
        # ── phoneme converter (lazy, one per CUDA device) ──────────────────
        self.language        = language
        self._t2p_cache: dict[str, Text2PhonemeSequence] = {}
        
        
     # ------------------------------------------------------------------- #
    # get Text2PhonemeSequence that lives on *device*                     #
    # ------------------------------------------------------------------- #
    def _get_t2p_on(self, device: torch.device) -> "Text2PhonemeSequence":
        key = str(device)
        if key not in self._t2p_cache:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    self._t2p_cache[key] = Text2PhonemeSequence(
                        language=self.language,
                        is_cuda=True,
                    )
            else:  # CPU
                self._t2p_cache[key] = Text2PhonemeSequence(
                    language=self.language,
                    is_cuda=False,
                )
        return self._t2p_cache[key]


    # ───────────────────────────────────────────────────────────────────────────
    # helpers
    # ───────────────────────────────────────────────────────────────────────────
    def _basic_clean(self, s: str) -> str:
        """Lower-case & remove punctuation (XPhoneBERT is case-insensitive)."""
        punc = f"[{re.escape(string.punctuation)}]"
        text_clean =  re.sub(punc, "", s.lower()).strip()
        # remove all double spaces
        text_clean = re.sub(r"\s+", " ", text_clean)
        # remove all spaces at the beginning and end
        text_clean = text_clean.strip()
        return text_clean

    def decode_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    # ────────────────────────────────────────────────────────────────
    #  TextGrid ↔ phoneme alignment (adapted to accept tg object)    #
    # ────────────────────────────────────────────────────────────────

    def align_phonemes_from_textgrid(
        self,
        phoneme_list: List[str],
        tg: tgt.core.TextGrid,
        add_sil: bool = False,
        relative_time: bool = False,
    ) -> List[Tuple[str, str, List[float]]]:
        """Align *phoneme_list* to a pre-loaded TextGrid `tg`."""
        global_xmin = tg.tiers[0].start_time
        global_xmax = tg.tiers[0].end_time

        word_items = [
            (str(i), {"xmin": iv.start_time, "xmax": iv.end_time, "text": iv.text})
            for i, iv in enumerate(tg.get_tier_by_name("words").intervals, start=1)
        ]
        phone_items = [
            (str(i), {"xmin": iv.start_time, "xmax": iv.end_time, "text": iv.text})
            for i, iv in enumerate(tg.get_tier_by_name("phones").intervals, start=1)
        ]
        word_items.sort(key=lambda x: int(x[0]))
        phone_items.sort(key=lambda x: int(x[0]))

        words_texts = [info["text"] for _, info in word_items]
        word_ranges = [(info["xmin"], info["xmax"]) for _, info in word_items]
        phone_infos = [info for _, info in phone_items]

        phone_groups: List[List[Dict[str, Any]]] = []
        pi_global = 0
        for wmin, wmax in word_ranges:
            grp: List[Dict[str, Any]] = []
            while pi_global < len(phone_infos):
                entry = phone_infos[pi_global]
                if wmin <= entry["xmin"] and entry["xmax"] <= wmax:
                    grp.append(entry)
                    pi_global += 1
                else:
                    break
            phone_groups.append(grp)

        aligned: List[Tuple[str, str, List[float]]] = []
        word_idx = phon_idx = 0
        n_words = len(word_items)

        for tok in phoneme_list:
            if tok == "▁":
                word_idx += 1
                phon_idx = 0
                val = " " if 0 <= word_idx < n_words and words_texts[word_idx] == "" else "NA"
                ts: List[float] = []
            else:
                grp = phone_groups[word_idx] if 0 <= word_idx < n_words else []
                entry = None
                if phon_idx < len(grp) and (
                    tok == grp[phon_idx]["text"]
                    or tok.lstrip("ˈˌ") == grp[phon_idx]["text"].lstrip("ˈˌ")
                ):
                    entry = grp[phon_idx]
                    phon_idx += 1
                elif phon_idx < len(grp):
                    entry = grp[phon_idx]
                    phon_idx = min(phon_idx + 1, len(grp) - 1)

                if entry:
                    val = entry["text"]
                    ts = [entry["xmin"], entry["xmax"]]
                else:
                    val, ts = "_", []

            aligned.append((tok, val, ts))

        if add_sil and phone_infos:
            first, last = phone_infos[0], phone_infos[-1]
            sil: List[Tuple[str, str, List[float]]] = []
            if first["xmin"] > global_xmin:
                sil.append(("▁", " ", [global_xmin, first["xmin"]]))
            if last["xmax"] < global_xmax:
                sil.append(("▁", " ", [last["xmax"], global_xmax]))
            aligned = sil[:1] + aligned + sil[1:]

        if relative_time:
            if global_xmax <= 0:
                raise ValueError("global_xmax must be > 0 for relative_time")
            aligned = [
                (tok, val, [t / global_xmax for t in ts]) if ts else (tok, val, [])
                for tok, val, ts in aligned
            ]
        return aligned


    # ───────────────────────────────────────────────────────────────────────────
    # forward
    # ───────────────────────────────────────────────────────────────────────────
    # def forward(self, sentences: List[str]
    #             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    def forward(self, sentences: List[str], tg_list: List[Any] | None = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    


        if not sentences:
            raise ValueError("input list is empty")

        # # inputs = [self._basic_clean(s) for s in sentences]
        
        # # Convert raw text → phoneme sequence (space-separated)
        # # inputs = [self.t2p.infer_sentence(self._basic_clean(s))
        # #           for s in sentences]                      # NEW / REPLACES OLD
        
        # ### NEW
        # # inputs, aligned_all = [], []
        # inputs, aligned_all, inp_lens = [], [], []      # NEW
        
        
        device = next(self.parameters()).device
        t2p    = self._get_t2p_on(device)               # ← NEW

        inputs, aligned_all, inp_lens = [], [], []
        
        
        tg_list = tg_list or [None] * len(sentences)
        for s, tg in zip(sentences, tg_list):
            # phon_str = self.t2p.infer_sentence(self._basic_clean(s))
            cleaned = self._basic_clean(s)
            # ── new: protect Text2PhonemeSequence from empty strings ──────────
            if cleaned == "":
                # Treat it as “silence”: a single space → one phoneme “▁”
                phon_str = "▁"
                print(f"[DEBUG] '{s}' became empty after cleaning → "
                    f"using placeholder phoneme '▁'")
            else:
                try:
                    phon_str = t2p.infer_sentence(cleaned)
                
                except Exception as e:
                    # fall back gracefully instead of crashing the whole batch
                    print(f"[WARN] Text-to-phoneme failed for '{s}': {e}")
                    phon_str = "▁"
        
            inputs.append(phon_str)
            inp_lens.append(len(phon_str.split()))      # NEW
            if tg is not None:
                aligned_all.append(
                    # self.align_phonemes_from_textgrid(phon_str.split(), tg)
                    self.align_phonemes_from_textgrid(phon_str.split(), tg, add_sil=True, relative_time=True)
                )
            else:
                aligned_all.append(None)

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
        
        # ── keep only coordinates & sanity-check length ────────────────────
        # aligned_coords: List[List[List[float]]] = []
        T = seq_emb.shape[1]                      # same T for all items (pad)
        # for i, a in enumerate(aligned_all):
        #     if a is not None and len(a) == T:
                
        aligned_coords: List[List[List[float]]] = []
        for i, (a, L_exp) in enumerate(zip(aligned_all, inp_lens)):   # NEW
            if a is not None and len(a) == L_exp + 2:                     # FIX; +2 for incl. end and start tokens
                aligned_coords.append([t[2] for t in a])     # keep [xmin,xmax]
            else:
                if a is not None:                            # length mismatch
                    # print(f"[DEBUG] align mismatch @{i}: "
                    #       f"seq_len={T}  aligned_len={len(a)}")
                    print(f"[DEBUG] align mismatch @{i}: "
                          f"expected_len={L_exp}  aligned_len={len(a)}")
                    
                    
                    if abs(len(a) - (L_exp + 2)) < 3:
                        print('[DEBUG] len(a) is close to L_exp')
                        aligned_coords.append([t[2] for t in a])     # keep [xmin,xmax]
                    else:  
                       aligned_coords.append([[0.0, 0.0]] * L_exp)  # zero-coords
                       print(f"[DEBUG] big gap in length @{i}: "
                             f"expected_len={L_exp}  aligned_len={len(a)}")
                       print(f"aligned_all[{i}] = {a}")
                       
                

        # return global_emb, seq_emb, key_mask, aligned_all   # NEW
        return global_emb, seq_emb, key_mask, aligned_coords   # NEW



# ──────────────────────────────────────────────────────────────────────────────
# debug / smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = [
        "Please call Stella.",
        "Ask her to bring these things with her from the store.",
        "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
        "We also need a small plastic snake and a big toy frog for the kids.",
        "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.",
        "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
        "The rainbow is a division of white light into many beautiful colors.",
        "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon.",
        "There is , according to legend, a boiling pot of gold at one end.",
        "People look, but no one ever finds it."        
    ]
    
    device  = torch.device("cuda", 0)           # or 1, 2 … as needed
    encoder = TextEncoder(hidden_dim=256, freeze_backbone=True).eval().to(device)

    with torch.no_grad():
        for sent in sample:
            cleaned   = encoder._basic_clean(sent)
            # phonemes  = encoder.t2p.infer_sentence(cleaned)
            phonemes = encoder._get_t2p_on(device).infer_sentence(cleaned)
            ids       = encoder.tokenizer(phonemes,
                                          add_special_tokens=True)["input_ids"]
            tokens    = encoder.tokenizer.convert_ids_to_tokens(ids)

            print("-" * 60)
            print("CLEANED   :", cleaned)
            print("PHONEMES :", phonemes)
            print("IDS      :", ids)
            print("TOKENS   :", tokens)

        # run a forward pass to ensure nothing breaks
        
        # ------------------------------- demo with TextGrids ------------------ #
        tg_paths = [
            "data/voicebank_demand/textgrids_ipa/val/p226_001.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_002.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_003.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_004.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_005.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_006.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_007.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_008.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_009.TextGrid",
            "data/voicebank_demand/textgrids_ipa/val/p226_010.TextGrid",        
        ]
        tg_objs = [tgt.read_textgrid(p) for p in tg_paths]
    
        # run a forward pass with alignment
        g, s, m, aligned = encoder(sample, tg_list=tg_objs)
        print("-" * 60)
        for i, a in enumerate(aligned):
            print(f"[{i}] aligned len={len(a) if a else 0}")
            g_len = g[i].shape[0]           # hidden-dim (global vector length)
            s_len = s[i].shape[0]           # token count  (seq_emb / key_mask)
            m_len = m[i].shape[0]
            print(f"g_len={g_len}  s_len={s_len}  m_len={m_len}")
            print(f"aligned: {a}")
            # print(f'key_mask: {m[i]}')
        
        # print length of g, s, m in each batch
        print(f"g: {g.shape}, s: {s.shape}, m: {m.shape}")    
            
        print("-" * 60)
        print("Forward pass OK ✔")