import os
import yaml
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer
import sys
import re, string                                          # ADD 02 MAY



### 04 MAY - FIX: make it work with newer torch versions
# ─── Allow the custom class stored in the PL‑BERT checkpoint ──────────────────
from torch.serialization import add_safe_globals
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer
add_safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer])
# ───────────────────────────────────────────────────────────────────────────────
### 04 MAY - FIX: make it work with newer torch versions

# Allow remote code if needed
os.environ["TRUST_REMOTE_CODE"] = "True"

# Define the PLBERT path relative to this file
PLBERT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../_miipher/miipher2.0/plbert/'))
if PLBERT_PATH not in sys.path:
    sys.path.insert(0, PLBERT_PATH)

# Instead of using EspeakBackend, we use OpenPhonemizer
from openphonemizer import OpenPhonemizer
from text_utils import TextCleaner

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim, seq_dim=None, freeze_plbert=True):
        """
        Args:
            hidden_dim (int): Dimension of the global output embedding (for FiLM).
            seq_dim (int, optional): Dimension of the sequence embeddings (for cross-attention).
                                     Defaults to hidden_dim if not provided.
            freeze_plbert (bool): Whether to freeze PL-BERT weights during training.
        """
        super(TextEncoder, self).__init__()
        plbert_root = PLBERT_PATH
        log_dir = os.path.join(plbert_root, "Checkpoint")
        config_path = os.path.join(log_dir, "config.yml")
        plbert_config = yaml.safe_load(open(config_path, "r"))
        albert_config = AlbertConfig(**plbert_config['model_params'])
        self.plbert = AlbertModel(albert_config)
        
        # Load the latest checkpoint from PL-BERT
        ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("step_")]
        iters = sorted([int(f.split('_')[-1].split('.')[0]) for f in ckpt_files])[-1]
        checkpoint_path = os.path.join(log_dir, f"step_{iters}.t7")
        print("Loading PL-BERT checkpoint from:", checkpoint_path)
        # checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) ### 04 MAY - FIX: make it work with newer torch versions
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.'
            if name.startswith('encoder.'):
                name = name[8:]  # remove 'encoder.'
                new_state_dict[name] = v
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in self.plbert.state_dict()}
        self.plbert.load_state_dict(filtered_state_dict, strict=False)
        
        # Instantiate the tokenizer from PL-BERT config
        self.tokenizer = TransfoXLTokenizer.from_pretrained(plbert_config['dataset_params']['tokenizer'])
        self.text_cleaner = TextCleaner()

                # --- after self.text_cleaner = TextCleaner() ---
        def _find_phon_vocab(obj):
            """Recursively search for an attr whose length matches
            the largest token id we saw during a quick test run."""
            # Most common attribute names in PL-BERT repos
            CANDIDATE_ATTRS = ["idx2token", "index2token", "itos",
                            "token_list", "vocab", "symbols",
                            "symbols_list", "phonemes"]

            visited = set()
            stack = [obj]
            while stack:
                cur = stack.pop()
                if id(cur) in visited:
                    continue
                visited.add(id(cur))

                for name in dir(cur):
                    if name.startswith("_"):
                        continue
                    try:
                        val = getattr(cur, name)
                    except Exception:
                        continue

                    if isinstance(val, (list, tuple)) and len(val) > 100:
                        # plausible vocab object
                        return val
                    if isinstance(val, dict) and all(isinstance(k, int) for k in val):
                        # dict mapping id->symbol
                        return val
                    if any(key in name.lower() for key in CANDIDATE_ATTRS):
                        return val

                    # keep digging
                    if not isinstance(val, (int, float, str, bytes)):
                        stack.append(val)
            return None

        self.id2phon = _find_phon_vocab(self.text_cleaner)
        if self.id2phon is None:
            print("[WARNING] Phoneme vocabulary not found – debug prints will be wrong")



        try:
            # TextCleaner usually owns or wraps the SequenceTokenizer
            #   └── self.text_cleaner.preproc.tokenizer  (PL-BERT original code)
            self.phon_tokenizer = self.text_cleaner.preproc.tokenizer
        except AttributeError:
            # fall-backs in case the codebase changed
            if hasattr(self.text_cleaner, "tokenizer"):
                self.phon_tokenizer = self.text_cleaner.tokenizer
            elif hasattr(self.text_cleaner, "token2idx"):      # very old versions
                from types import SimpleNamespace
                self.phon_tokenizer = SimpleNamespace(
                    token2idx=self.text_cleaner.token2idx
                )
            else:
                self.phon_tokenizer = None        # will trigger the old behaviour


        if self.phon_tokenizer is not None:
            self.idx2phon = (
                self.phon_tokenizer.idx2token       # many repos expose this list
                if hasattr(self.phon_tokenizer, "idx2token")
                else {i: t for t, i in self.phon_tokenizer.token2idx.items()}
            )


        
        # Initialize OpenPhonemizer (no external dependency on espeak)
        self.phonemizer = OpenPhonemizer()
        
        # Projection layers for global and sequence outputs.
        self.fc_global = nn.Linear(self.plbert.config.hidden_size, hidden_dim)
        if seq_dim is None:
            seq_dim = hidden_dim
        self.fc_seq = nn.Linear(self.plbert.config.hidden_size, seq_dim)
        
        if freeze_plbert:
            for param in self.plbert.parameters():
                param.requires_grad = False

        # After defining self.fc_global and self.fc_seq
        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.xavier_uniform_(self.fc_seq.weight)
        nn.init.zeros_(self.fc_global.bias)
        nn.init.zeros_(self.fc_seq.bias)

        # Cache the tokenizer
        self.phoneme_cache = {}
        
        
        ### DEBUGGING ###
        # Print sample of vocabulary to understand token mappings
        try:
            if hasattr(self.tokenizer, 'vocab'):
                vocab = self.tokenizer.vocab
                print(f"[DEBUG] Vocabulary size: {len(vocab)}")
                print(f"[DEBUG] Sample vocabulary items (first 10):")
                for i, (token, idx) in enumerate(list(vocab.items())[:10]):
                    print(f"  - Token: '{token}', ID: {idx}")
            else:
                print("[DEBUG] No directly accessible vocabulary found in tokenizer")
        except Exception as e:
            print(f"[WARNING] Error accessing vocabulary: {e}")
        ### DEBUGGING ###


        ## ADD 08 MAY ##
        # Add layer normalization before projection to stabilize embeddings
        self.seq_norm = nn.LayerNorm(self.plbert.config.hidden_size)
        self.global_norm = nn.LayerNorm(self.plbert.config.hidden_size)
        ## ADD 08 MAY ##        

    # ### DEBUGGING ###
    # def decode_tokens(self, token_ids):
    #     """Helper method to decode token IDs back to their token representations."""
    #     try:
    #         if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
    #             return self.tokenizer.convert_ids_to_tokens(token_ids)
    #         else:
    #             return [f"<ID:{i}>" for i in token_ids]  # Fallback if method not available
    #     except Exception as e:
    #         print(f"[WARNING] Error decoding tokens: {e}")
    #         return [f"<ERROR:{i}>" for i in token_ids]
    # ### DEBUGGING ###    

    ### DEBUGGING2 ###
    def decode_tokens(self, ids):
        # Primary: phoneme vocabulary we just found
        if hasattr(self, "id2phon") and self.id2phon is not None:
            if isinstance(self.id2phon, (list, tuple)):
                return [self.id2phon[i] if i < len(self.id2phon) else f"<UNK:{i}>" for i in ids]
            else:  # dict
                return [self.id2phon.get(i, f"<UNK:{i}>") for i in ids]

        # Fallbacks (old behaviour)
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return self.tokenizer.convert_ids_to_tokens(ids)
        return [f"<ID:{i}>" for i in ids]

    ### DEBUGGING2 ###



    def tokenize(self, sents):
        batched = []
        max_len = 0
        
        # Add debug print to show original sentences
        print(f"[DEBUG] Original sentences: {sents}")
        
        for sent in sents:

            # Handle empty strings
            if not sent or not sent.strip():
                # Use a placeholder token or set of tokens instead of empty
                token_ids = torch.LongTensor([0])  # Placeholder token ID
                batched.append(token_ids)
                max_len = max(max_len, 1)
                print(f"[DEBUG] Empty string detected, using placeholder token ID: [0]")
                continue

            # Use OpenPhonemizer to get phonemes.
            # Assume it returns a string; adjust if a list is returned.
            try:
                if sent in self.phoneme_cache:
                    phonemes = self.phoneme_cache[sent]
                else:
                    phonemes = self.phonemizer(sent)
                    self.phoneme_cache[sent] = phonemes
                print(f"[DEBUG] Phonemized: '{sent}' -> '{phonemes}'")
            except Exception as e:
                print(f"[WARNING] Phonemization failed for: '{sent}'. Error: {e}")
                phonemes = sent  # Fallback to the original text
                
            # If needed, you could split on whitespace: pretext = ' '.join(phonemes)
            pretext = phonemes
            cleaned = self.text_cleaner(pretext)
            
             # Debug print for cleaned data
            print(f"[DEBUG] After text_cleaner: {cleaned[:10]}... (showing first 10 elements)")
            
            # Assume text_cleaner returns a list of token IDs
            token_ids = torch.LongTensor(cleaned)
            
            
            # Print token IDs and try to decode them
            if len(token_ids) > 0:
                print(f"[DEBUG] Token IDs sample: {token_ids[:10].tolist()}... (Length: {len(token_ids)})")
                try:
                    tokens = self.decode_tokens(token_ids[:10].tolist())
                    print(f"[DEBUG] Decoded tokens sample: {tokens}")
                except Exception as e:
                    print(f"[WARNING] Could not decode tokens: {e}")
            
            
            batched.append(token_ids)
            max_len = max(max_len, len(token_ids))
        phoneme_ids = torch.zeros((len(sents), max_len), dtype=torch.long)
        mask = torch.zeros((len(sents), max_len), dtype=torch.float)
        for i, tokens in enumerate(batched):
            phoneme_ids[i, :len(tokens)] = tokens
            mask[i, :len(tokens)] = 1
            
        # Debug print final shapes and sample data
        print(f"[DEBUG] Final phoneme_ids shape: {phoneme_ids.shape}")
        if len(sents) > 0:
            active_tokens = (mask[0] == 1).sum().item()
            print(f"[DEBUG] First example active tokens: {active_tokens}")
            print(f"[DEBUG] First example token IDs: {phoneme_ids[0, :active_tokens].tolist()}")    
            
            
        return phoneme_ids, mask

    def forward(self, input_data):
        """
        Args:
            input_data (list of str): List of sentences.
        Returns:
            global_emb: Tensor of shape (B, hidden_dim)
            seq_emb: Tensor of shape (B, seq_len, seq_dim)
        """
        # print(f"[DEBUG TE] Processing text: type={type(input_data)}, len={len(input_data) if isinstance(input_data, list) else 'N/A'}")
        
        # ADD 02 MAY - text cleaning
        # input_data = [re.sub(f'[{re.escape(string.punctuation)}]',  # remove punctuation
        #                     '', sent.lower())                      # & lowercase
        #             for sent in input_data]
        
        # # remvoe spaces at the beginning and end of each sentence
        # input_data = [sent.strip() for sent in input_data]
        
        punct_re = re.compile(f"[{re.escape(string.punctuation)}]") # pre‑compiled once
        input_data = [punct_re.sub("", sent.lower()).strip()        # ↓ strip trims outer spaces
                    for sent in input_data]
        # print(f"[DEBUG TE] Processed text: {input_data}")    
        
        # ADD 02 MAY - text cleaning
        
        # Debug print for processed input before tokenization
        print(f"[DEBUG] Preprocessed input: {input_data[:2]}... (showing up to first 2 items)")
         
         
        
        
        phoneme_ids, attention_mask = self.tokenize(input_data)
        # print(f"[DEBUG] Phoneme IDs first example: {phoneme_ids[0].tolist()}")
        # print(f"[DEBUG] Attention mask first example: {attention_mask[0].tolist()}")
        print(f"[DEBUG] Non-zero tokens per example: {attention_mask.sum(dim=1).tolist()}")
        print(f"[DEBUG] Max sequence length: {phoneme_ids.shape[1]}")
        print(f"[DEBUG] Padding percentage: {(1 - attention_mask.float().mean()).item()*100:.2f}%")
        
        
        # Deeper token inspection for first example
        if phoneme_ids.shape[0] > 0:
            active_len = int(attention_mask[0].sum().item())
            decoded = self.decode_tokens(phoneme_ids[0, :active_len].tolist())
            print(f"[DEBUG] First example decoded tokens: {decoded}")
        
        
        # Move token IDs and attention_mask to the same device as PL-BERT:
        device = next(self.plbert.parameters()).device
        phoneme_ids = phoneme_ids.to(device)
        attention_mask = attention_mask.to(device)

        # In TextEncoder's forward method
        print(f"[DEBUG] Sample input text: '{input_data[0]}'")
        print(f"[DEBUG] Phonemized form: '{self.phonemizer(input_data[0])}'")
        print(f"[DEBUG] Token IDs shape: {phoneme_ids.shape}")
        print(f"[DEBUG] Active tokens: {attention_mask.sum(dim=1).tolist()}")
        

                
        outputs = self.plbert(phoneme_ids, attention_mask=attention_mask)
        # Use first token (CLS) for global embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]


        # global_emb = self.fc_global(cls_embedding)
        # # Project each token representation for sequence embedding
        # seq_emb = self.fc_seq(outputs.last_hidden_state)


        ## UPD 08 MAY ##
        global_emb = self.fc_global(self.global_norm(cls_embedding))
        seq_emb = self.fc_seq(self.seq_norm(outputs.last_hidden_state))
        # UPD 08 MAY ##
        
        # Build a boolean mask for cross-attention: True => mask out
        # if attention_mask is 1=real, 0=pad => we invert
        # i.e. text_key_mask[b, i] = (attention_mask[b, i] == 0)
        text_key_mask = (attention_mask == 0)
        # print(f"[DEBUG] Text key mask tokens: {text_key_mask.sum(dim=1).tolist()}")
        
        
        # Log embedding statistics for debugging
        print(f"[DEBUG] Global embedding stats - min: {global_emb.min().item():.3f}, max: {global_emb.max().item():.3f}, mean: {global_emb.mean().item():.3f}")
        
    
        return global_emb, seq_emb, text_key_mask
