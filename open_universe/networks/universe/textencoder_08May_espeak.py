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

# Import using absolute paths
import importlib.util
import os

# Import phonemize and other modules from PLBERT_PATH
phonemize_path = os.path.join(PLBERT_PATH, "phonemize.py")
text_utils_path = os.path.join(PLBERT_PATH, "text_utils.py")
text_normalize_path = os.path.join(PLBERT_PATH, "text_normalize.py")

# Load modules dynamically
spec = importlib.util.spec_from_file_location("phonemize", phonemize_path)
phonemize_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phonemize_module)
phonemize = phonemize_module.phonemize

spec = importlib.util.spec_from_file_location("text_utils", text_utils_path)
text_utils_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_utils_module)
TextCleaner = text_utils_module.TextCleaner

from phonemizer.backend import EspeakBackend

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
        
        # Initialize Espeak phonemizer
        self.phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        
        # Create our custom phoneme tokenizer
        self.create_phoneme_tokenizer()
        
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
                
                # Check if vocabulary contains phoneme characters
                phoneme_chars = {'ˈ', 'ɪ', 'ə', 'æ', 'ɑ', 'ɔ', 'ɛ', 'ɝ', 'ɚ', 'ʊ', 'ʌ', 'ð', 'θ', 'ʃ', 'ʒ', 'ŋ'}
                has_phoneme_tokens = any(any(c in token for c in phoneme_chars) for token in vocab.keys())
                print(f"[DEBUG] Tokenizer has phoneme tokens: {has_phoneme_tokens}")
                
                # Sample tokens that might be phonemes
                phoneme_like = [token for token in vocab.keys() if any(c in token for c in phoneme_chars)][:10]
                print(f"[DEBUG] Potential phoneme tokens: {phoneme_like}")
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

    def create_phoneme_tokenizer(self):
        """Create a simple mapping from phoneme characters to token IDs."""
        # Common phoneme symbols
        phoneme_chars = [
            'ˈ', 'ˌ', 'æ', 'ɑ', 'ɔ', 'ɛ', 'ɪ', 'ɝ', 'ɚ', 'ʊ', 'ʌ', 'ə', 
            'i', 'u', 'e', 'o', 'a', 'p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 
            'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'r', 'w', 'j', 
            'ː', ' ', 'ɐ', 'ɒ', 'ɜ', 'ɡ', 'ɹ', 'ʰ', 'ʲ', 'ʷ', 'ʔ'
        ]
        
        # Add all ASCII letters for fallback
        for c in string.ascii_lowercase:
            if c not in phoneme_chars:
                phoneme_chars.append(c)
        
        # Create mapping (reserve 0 for padding)
        self.phoneme_to_id = {ph: i+1 for i, ph in enumerate(phoneme_chars)}
        # Add special tokens
        self.phoneme_to_id['<PAD>'] = 0
        self.phoneme_to_id['<UNK>'] = len(self.phoneme_to_id)
        
        # Create reverse mapping
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
        
        print(f"[DEBUG] Created phoneme tokenizer with {len(self.phoneme_to_id)} tokens")
        return self.phoneme_to_id

    ### DEBUGGING ###
    def decode_tokens(self, token_ids):
        """Helper method to decode token IDs back to their token representations."""
        try:
            # If we have custom phoneme tokenizer, use it
            if hasattr(self, 'id_to_phoneme'):
                return [self.id_to_phoneme.get(id, '<UNK>') for id in token_ids]
                
            if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
                return self.tokenizer.convert_ids_to_tokens(token_ids)
            else:
                return [f"<ID:{i}>" for i in token_ids]  # Fallback if method not available
        except Exception as e:
            print(f"[WARNING] Error decoding tokens: {e}")
            return [f"<ERROR:{i}>" for i in token_ids]
    ### DEBUGGING ###    

    def tokenize(self, sents):
        """Updated tokenization logic using direct phoneme-to-ID mapping."""
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

            # Use the phonemize function from PL-BERT
            try:
                if sent in self.phoneme_cache:
                    phonemized_result = self.phoneme_cache[sent]
                else:
                    phonemized_result = phonemize(sent, self.phonemizer, self.tokenizer)
                    self.phoneme_cache[sent] = phonemized_result
                
                # Get phonemes from the result and join with spaces
                phonemes = ' '.join(phonemized_result['phonemes'])
                print(f"[DEBUG] Phonemized: '{sent}' -> '{phonemes}'")
                print(f"[DEBUG] phonemized_result keys: {phonemized_result.keys()}")
                
                # For debugging, also show the raw phoneme list
                print(f"[DEBUG] Raw phoneme list: {phonemized_result['phonemes']}")
            except Exception as e:
                print(f"[WARNING] Phonemization failed for: '{sent}'. Error: {e}")
                phonemes = sent  # Fallback to the original text
            
            # Use direct phoneme-to-id mapping
            direct_tokens = []
            for char in phonemes:
                direct_tokens.append(self.phoneme_to_id.get(char, self.phoneme_to_id['<UNK>']))
            
            # Print token IDs from direct mapping
            print(f"[DEBUG] Direct phoneme tokenization: {direct_tokens[:10]}... (Length: {len(direct_tokens)})")
            direct_token_ids = torch.LongTensor(direct_tokens)
            
            # For comparison, also show the original method with text_cleaner
            print(f"[DEBUG] Raw phonemes before text_cleaner: '{phonemes}'")
            cleaned = self.text_cleaner(phonemes)
            print(f"[DEBUG] After text_cleaner: {cleaned[:10]}... (showing first 10 elements)")
            text_cleaner_ids = torch.LongTensor(cleaned)
            print(f"[DEBUG] text_cleaner IDs: {text_cleaner_ids[:10].tolist()}... (Length: {len(text_cleaner_ids)})")
            
            # Try to decode both tokenization methods
            print(f"[DEBUG] Decoded direct tokens: {self.decode_tokens(direct_tokens[:10])}")
            
            # Use the direct phoneme tokenization
            batched.append(direct_token_ids)
            max_len = max(max_len, len(direct_token_ids))
        
        # Create padded tensors
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
            print(f"[DEBUG] First example decoded: {''.join(self.decode_tokens(phoneme_ids[0, :min(20, active_tokens)].tolist()))}")
        
        return phoneme_ids, mask

    def forward(self, input_data):
        """
        Args:
            input_data (list of str): List of sentences.
        Returns:
            global_emb: Tensor of shape (B, hidden_dim)
            seq_emb: Tensor of shape (B, seq_len, seq_dim)
        """
        # Text cleaning (same as before)
        punct_re = re.compile(f"[{re.escape(string.punctuation)}]") # pre-compiled once
        input_data = [punct_re.sub("", sent.lower()).strip()        # strip trims outer spaces
                    for sent in input_data]
        
        # Debug print for processed input before tokenization
        print(f"[DEBUG] Preprocessed input: {input_data[:2]}... (showing up to first 2 items)")
        
        # Tokenize the input
        phoneme_ids, attention_mask = self.tokenize(input_data)
        
        # Debug prints
        print(f"[DEBUG] Non-zero tokens per example: {attention_mask.sum(dim=1).tolist()}")
        print(f"[DEBUG] Max sequence length: {phoneme_ids.shape[1]}")
        print(f"[DEBUG] Padding percentage: {(1 - attention_mask.float().mean()).item()*100:.2f}%")
        
        # Deeper token inspection for first example
        if phoneme_ids.shape[0] > 0:
            active_len = int(attention_mask[0].sum().item())
            decoded = self.decode_tokens(phoneme_ids[0, :active_len].tolist())
            print(f"[DEBUG] First example decoded tokens: {decoded}")
        
        # Move token IDs and attention_mask to the same device as PL-BERT
        device = next(self.plbert.parameters()).device
        phoneme_ids = phoneme_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Additional debugging
        if len(input_data) > 0:
            # Get phonemized form for display
            phonemized_sample = phonemize(input_data[0], self.phonemizer, self.tokenizer)
            phonemes_str = ' '.join(phonemized_sample['phonemes'])
            
            print(f"[DEBUG] Sample input text: '{input_data[0]}'")
            print(f"[DEBUG] Phonemized form: '{phonemes_str}'")
            print(f"[DEBUG] Token IDs shape: {phoneme_ids.shape}")
            print(f"[DEBUG] Active tokens: {attention_mask.sum(dim=1).tolist()}")
        
        # Forward pass through PLBERT
        outputs = self.plbert(phoneme_ids, attention_mask=attention_mask)
        
        # Use first token (CLS) for global embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Apply normalization and projection
        global_emb = self.fc_global(self.global_norm(cls_embedding))
        seq_emb = self.fc_seq(self.seq_norm(outputs.last_hidden_state))
        
        # Build a boolean mask for cross-attention: True => mask out
        text_key_mask = (attention_mask == 0)
        
        # Log embedding statistics for debugging
        print(f"[DEBUG] Global embedding stats - min: {global_emb.min().item():.3f}, max: {global_emb.max().item():.3f}, mean: {global_emb.mean().item():.3f}")
        
        return global_emb, seq_emb, text_key_mask


# import os
# import yaml
# from collections import OrderedDict
# import torch
# import torch.nn as nn
# from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer
# import sys
# import re, string                                          # ADD 02 MAY

# ### 04 MAY - FIX: make it work with newer torch versions
# # ─── Allow the custom class stored in the PL‑BERT checkpoint ──────────────────
# from torch.serialization import add_safe_globals
# from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer
# add_safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer])
# # ───────────────────────────────────────────────────────────────────────────────
# ### 04 MAY - FIX: make it work with newer torch versions

# # Allow remote code if needed
# os.environ["TRUST_REMOTE_CODE"] = "True"

# # Define the PLBERT path relative to this file
# PLBERT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../_miipher/miipher2.0/plbert/'))
# if PLBERT_PATH not in sys.path:
#     sys.path.insert(0, PLBERT_PATH)

# # Import from PLBERT instead of using OpenPhonemizer
# from phonemizer.backend import EspeakBackend
# # from plbert.phonemize import phonemize
# # from plbert.text_utils import TextCleaner
# # from plbert.text_normalize import normalize_text, remove_accents



# # Import using absolute paths
# import importlib.util
# import os
 
# # Import phonemize and other modules from PLBERT_PATH
# phonemize_path = os.path.join(PLBERT_PATH, "phonemize.py")
# text_utils_path = os.path.join(PLBERT_PATH, "text_utils.py")
# text_normalize_path = os.path.join(PLBERT_PATH, "text_normalize.py")

# # Load modules dynamically
# spec = importlib.util.spec_from_file_location("phonemize", phonemize_path)
# phonemize_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(phonemize_module)
# phonemize = phonemize_module.phonemize

# spec = importlib.util.spec_from_file_location("text_utils", text_utils_path)
# text_utils_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(text_utils_module)
# TextCleaner = text_utils_module.TextCleaner



# class TextEncoder(nn.Module):
#     def __init__(self, hidden_dim, seq_dim=None, freeze_plbert=True):
#         """
#         Args:
#             hidden_dim (int): Dimension of the global output embedding (for FiLM).
#             seq_dim (int, optional): Dimension of the sequence embeddings (for cross-attention).
#                                      Defaults to hidden_dim if not provided.
#             freeze_plbert (bool): Whether to freeze PL-BERT weights during training.
#         """
#         super(TextEncoder, self).__init__()
#         plbert_root = PLBERT_PATH
#         log_dir = os.path.join(plbert_root, "Checkpoint")
#         config_path = os.path.join(log_dir, "config.yml")
#         plbert_config = yaml.safe_load(open(config_path, "r"))
#         albert_config = AlbertConfig(**plbert_config['model_params'])
#         self.plbert = AlbertModel(albert_config)
        
#         # Load the latest checkpoint from PL-BERT
#         ckpt_files = [f for f in os.listdir(log_dir) if f.startswith("step_")]
#         iters = sorted([int(f.split('_')[-1].split('.')[0]) for f in ckpt_files])[-1]
#         checkpoint_path = os.path.join(log_dir, f"step_{iters}.t7")
#         print("Loading PL-BERT checkpoint from:", checkpoint_path)
#         # checkpoint = torch.load(checkpoint_path, map_location="cpu")
#         checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) ### 04 MAY - FIX: make it work with newer torch versions
#         state_dict = checkpoint['net']
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = k[7:]  # remove 'module.'
#             if name.startswith('encoder.'):
#                 name = name[8:]  # remove 'encoder.'
#                 new_state_dict[name] = v
#         filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in self.plbert.state_dict()}
#         self.plbert.load_state_dict(filtered_state_dict, strict=False)
        
#         # Instantiate the tokenizer from PL-BERT config
#         self.tokenizer = TransfoXLTokenizer.from_pretrained(plbert_config['dataset_params']['tokenizer'])
#         self.text_cleaner = TextCleaner()
        
#         # Initialize Espeak phonemizer instead of OpenPhonemizer
#         self.phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
        
#         # Projection layers for global and sequence outputs.
#         self.fc_global = nn.Linear(self.plbert.config.hidden_size, hidden_dim)
#         if seq_dim is None:
#             seq_dim = hidden_dim
#         self.fc_seq = nn.Linear(self.plbert.config.hidden_size, seq_dim)
        
#         if freeze_plbert:
#             for param in self.plbert.parameters():
#                 param.requires_grad = False

#         # After defining self.fc_global and self.fc_seq
#         nn.init.xavier_uniform_(self.fc_global.weight)
#         nn.init.xavier_uniform_(self.fc_seq.weight)
#         nn.init.zeros_(self.fc_global.bias)
#         nn.init.zeros_(self.fc_seq.bias)

#         # Cache the tokenizer
#         self.phoneme_cache = {}
        
#         ### DEBUGGING ###
#         # Print sample of vocabulary to understand token mappings
#         try:
#             if hasattr(self.tokenizer, 'vocab'):
#                 vocab = self.tokenizer.vocab
#                 print(f"[DEBUG] Vocabulary size: {len(vocab)}")
#                 print(f"[DEBUG] Sample vocabulary items (first 10):")
#                 for i, (token, idx) in enumerate(list(vocab.items())[:10]):
#                     print(f"  - Token: '{token}', ID: {idx}")
#             else:
#                 print("[DEBUG] No directly accessible vocabulary found in tokenizer")
#         except Exception as e:
#             print(f"[WARNING] Error accessing vocabulary: {e}")
#         ### DEBUGGING ###

#         ## ADD 08 MAY ##
#         # Add layer normalization before projection to stabilize embeddings
#         self.seq_norm = nn.LayerNorm(self.plbert.config.hidden_size)
#         self.global_norm = nn.LayerNorm(self.plbert.config.hidden_size)
#         ## ADD 08 MAY ##        

#     ### DEBUGGING ###
#     def decode_tokens(self, token_ids):
#         """Helper method to decode token IDs back to their token representations."""
#         try:
#             if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
#                 return self.tokenizer.convert_ids_to_tokens(token_ids)
#             else:
#                 return [f"<ID:{i}>" for i in token_ids]  # Fallback if method not available
#         except Exception as e:
#             print(f"[WARNING] Error decoding tokens: {e}")
#             return [f"<ERROR:{i}>" for i in token_ids]
#     ### DEBUGGING ###    

#     def tokenize(self, sents):
#         """Updated tokenization logic based on the reference implementation."""
#         batched = []
#         max_len = 0
        
#         # Add debug print to show original sentences
#         print(f"[DEBUG] Original sentences: {sents}")
        
#         for sent in sents:
#             # Handle empty strings
#             if not sent or not sent.strip():
#                 # Use a placeholder token or set of tokens instead of empty
#                 token_ids = torch.LongTensor([0])  # Placeholder token ID
#                 batched.append(token_ids)
#                 max_len = max(max_len, 1)
#                 print(f"[DEBUG] Empty string detected, using placeholder token ID: [0]")
#                 continue

#             # Use the phonemize function from PL-BERT
#             try:
#                 if sent in self.phoneme_cache:
#                     phonemized_result = self.phoneme_cache[sent]
#                 else:
#                     phonemized_result = phonemize(sent, self.phonemizer, self.tokenizer)
#                     self.phoneme_cache[sent] = phonemized_result
                
#                 # Get phonemes from the result and join with spaces as in the reference code
#                 phonemes = ' '.join(phonemized_result['phonemes'])
#                 print(f"[DEBUG] Phonemized: '{sent}' -> '{phonemes}'")
#             except Exception as e:
#                 print(f"[WARNING] Phonemization failed for: '{sent}'. Error: {e}")
#                 phonemes = sent  # Fallback to the original text
            
#             # Process with text_cleaner (matched to reference implementation)
#             cleaned = self.text_cleaner(phonemes)
            
#             # Debug print for cleaned data
#             print(f"[DEBUG] After text_cleaner: {cleaned[:10]}... (showing first 10 elements)")
            
#             # Create token IDs tensor
#             token_ids = torch.LongTensor(cleaned)
            
#             # Print token IDs and try to decode them
#             if len(token_ids) > 0:
#                 print(f"[DEBUG] Token IDs sample: {token_ids[:10].tolist()}... (Length: {len(token_ids)})")
#                 try:
#                     tokens = self.decode_tokens(token_ids[:10].tolist())
#                     print(f"[DEBUG] Decoded tokens sample: {tokens}")
#                 except Exception as e:
#                     print(f"[WARNING] Could not decode tokens: {e}")
            
#             batched.append(token_ids)
#             max_len = max(max_len, len(token_ids))
        
#         # Create padded tensors
#         phoneme_ids = torch.zeros((len(sents), max_len), dtype=torch.long)
#         mask = torch.zeros((len(sents), max_len), dtype=torch.float)
        
#         for i, tokens in enumerate(batched):
#             phoneme_ids[i, :len(tokens)] = tokens
#             mask[i, :len(tokens)] = 1
        
#         # Debug print final shapes and sample data
#         print(f"[DEBUG] Final phoneme_ids shape: {phoneme_ids.shape}")
#         if len(sents) > 0:
#             active_tokens = (mask[0] == 1).sum().item()
#             print(f"[DEBUG] First example active tokens: {active_tokens}")
#             print(f"[DEBUG] First example token IDs: {phoneme_ids[0, :active_tokens].tolist()}")
        
#         return phoneme_ids, mask

#     def forward(self, input_data):
#         """
#         Args:
#             input_data (list of str): List of sentences.
#         Returns:
#             global_emb: Tensor of shape (B, hidden_dim)
#             seq_emb: Tensor of shape (B, seq_len, seq_dim)
#         """
#         # Text cleaning (same as before)
#         punct_re = re.compile(f"[{re.escape(string.punctuation)}]") # pre-compiled once
#         input_data = [punct_re.sub("", sent.lower()).strip()        # strip trims outer spaces
#                     for sent in input_data]
        
#         # Debug print for processed input before tokenization
#         print(f"[DEBUG] Preprocessed input: {input_data[:2]}... (showing up to first 2 items)")
        
#         # Tokenize the input
#         phoneme_ids, attention_mask = self.tokenize(input_data)
        
#         # Debug prints
#         print(f"[DEBUG] Non-zero tokens per example: {attention_mask.sum(dim=1).tolist()}")
#         print(f"[DEBUG] Max sequence length: {phoneme_ids.shape[1]}")
#         print(f"[DEBUG] Padding percentage: {(1 - attention_mask.float().mean()).item()*100:.2f}%")
        
#         # Deeper token inspection for first example
#         if phoneme_ids.shape[0] > 0:
#             active_len = int(attention_mask[0].sum().item())
#             decoded = self.decode_tokens(phoneme_ids[0, :active_len].tolist())
#             print(f"[DEBUG] First example decoded tokens: {decoded}")
        
#         # Move token IDs and attention_mask to the same device as PL-BERT
#         device = next(self.plbert.parameters()).device
#         phoneme_ids = phoneme_ids.to(device)
#         attention_mask = attention_mask.to(device)
        
#         # Additional debugging
#         if len(input_data) > 0:
#             # Get phonemized form for display
#             phonemized_sample = phonemize(input_data[0], self.phonemizer, self.tokenizer)
#             phonemes_str = ' '.join(phonemized_sample['phonemes'])
            
#             print(f"[DEBUG] Sample input text: '{input_data[0]}'")
#             print(f"[DEBUG] Phonemized form: '{phonemes_str}'")
#             print(f"[DEBUG] Token IDs shape: {phoneme_ids.shape}")
#             print(f"[DEBUG] Active tokens: {attention_mask.sum(dim=1).tolist()}")
        
#         # Forward pass through PLBERT
#         outputs = self.plbert(phoneme_ids, attention_mask=attention_mask)
        
#         # Use first token (CLS) for global embedding
#         cls_embedding = outputs.last_hidden_state[:, 0, :]
        
#         # Apply normalization and projection
#         global_emb = self.fc_global(self.global_norm(cls_embedding))
#         seq_emb = self.fc_seq(self.seq_norm(outputs.last_hidden_state))
        
#         # Build a boolean mask for cross-attention: True => mask out
#         text_key_mask = (attention_mask == 0)
        
#         # Log embedding statistics for debugging
#         print(f"[DEBUG] Global embedding stats - min: {global_emb.min().item():.3f}, max: {global_emb.max().item():.3f}, mean: {global_emb.mean().item():.3f}")
        
#         return global_emb, seq_emb, text_key_mask