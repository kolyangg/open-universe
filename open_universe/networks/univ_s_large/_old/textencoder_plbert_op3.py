import os
import yaml
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer
import sys

# Allow remote code if needed
os.environ["TRUST_REMOTE_CODE"] = "True"

# # Define the PLBERT path relative to this file
# PLBERT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../_miipher/miipher2.0/plbert/'))
# if PLBERT_PATH not in sys.path:
#     sys.path.insert(0, PLBERT_PATH)
    
    
# In textencoder_plbert_op3.py, replace the PLBERT_PATH line with:

# Define the PLBERT path by checking environment variable first
if "PLBERT_PATH" in os.environ and os.path.exists(os.environ["PLBERT_PATH"]):
    PLBERT_PATH = os.environ["PLBERT_PATH"]
    print(f"Using PLBERT_PATH from environment: {PLBERT_PATH}")
else:
    # Fall back to relative path as before
    # PLBERT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../_miipher/miipher2.0/plbert/')) ### TEMP FOR NOTEBOOK!!! ###
    PLBERT_PATH = "/home/kolyangg/Dipl/speech_enh/_miipher/miipher2.0/plbert/"
    print(f"Using default PLBERT_PATH: {PLBERT_PATH}")

if PLBERT_PATH not in sys.path:
    sys.path.insert(0, PLBERT_PATH)

# Instead of using EspeakBackend, we use OpenPhonemizer
from openphonemizer import OpenPhonemizer
from text_utils import TextCleaner

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim, seq_dim=None, freeze_plbert=False): # Keep unfrozen to allow fine-tuning
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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

    def tokenize(self, sents):
        batched = []
        max_len = 0
        for sent in sents:

            # Handle empty strings
            if not sent or not sent.strip():
                # Create a meaningful phonetic representation instead of just [0]
                # This gives the model something to learn from even with empty text
                token_ids = self.text_cleaner("silence")
                batched.append(token_ids)
                max_len = max(max_len, len(token_ids))
                continue
            
            # Use OpenPhonemizer to get phonemes.
            # Assume it returns a string; adjust if a list is returned.
            try:
                if sent in self.phoneme_cache:
                    phonemes = self.phoneme_cache[sent]
                else:
                    phonemes = self.phonemizer(sent)
                    self.phoneme_cache[sent] = phonemes
            except Exception as e:
                print(f"[WARNING] Phonemization failed for: '{sent}'. Error: {e}")
                phonemes = sent  # Fallback to the original text
                
            # If needed, you could split on whitespace: pretext = ' '.join(phonemes)
            pretext = phonemes
            cleaned = self.text_cleaner(pretext)
            # Assume text_cleaner returns a list of token IDs
            token_ids = torch.LongTensor(cleaned)
            batched.append(token_ids)
            max_len = max(max_len, len(token_ids))
        phoneme_ids = torch.zeros((len(sents), max_len), dtype=torch.long)
        mask = torch.zeros((len(sents), max_len), dtype=torch.float)
        for i, tokens in enumerate(batched):
            phoneme_ids[i, :len(tokens)] = tokens
            mask[i, :len(tokens)] = 1
        return phoneme_ids, mask

    def forward(self, input_data):
        """
        Args:
            input_data (list of str): List of sentences.
        Returns:
            global_emb: Tensor of shape (B, hidden_dim)
            seq_emb: Tensor of shape (B, seq_len, seq_dim)
            attention_weights: Attention weights for interpretability (optional)
        """
        phoneme_ids, attention_mask = self.tokenize(input_data)
        # Move token IDs and attention_mask to the same device as PL-BERT:
        device = next(self.plbert.parameters()).device
        phoneme_ids = phoneme_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Debug information
        print(f"[DEBUG] Sample input text: '{input_data[0]}'")
        print(f"[DEBUG] Phonemized form: '{self.phonemizer(input_data[0])}'")
        print(f"[DEBUG] Token IDs shape: {phoneme_ids.shape}")
        print(f"[DEBUG] Active tokens: {attention_mask.sum(dim=1).tolist()}")
        
        # Add dropout to improve robustness (only during training)
        if self.training:
            # Apply token dropout - randomly mask 10% of tokens
            # This helps the model be robust to missing information
            token_mask = torch.bernoulli(torch.full_like(attention_mask, 0.9)).bool()
            # Don't drop CLS token (first token)
            token_mask[:, 0] = True
            # Only apply to tokens that were originally active
            token_mask = token_mask & attention_mask.bool()
            new_attention_mask = token_mask.float()
        else:
            new_attention_mask = attention_mask
                
        # Get PLBERT embeddings
        outputs = self.plbert(phoneme_ids, attention_mask=new_attention_mask)
        
        # Make a better global embedding by pooling important tokens
        # Use attention to identify important tokens rather than just CLS
        last_hidden = outputs.last_hidden_state
        
        # Use CLS token as query to attend to other tokens
        cls_embedding = last_hidden[:, 0, :].unsqueeze(1)  # [B, 1, hidden]
        
        # Simple attention mechanism
        attn_scores = torch.bmm(cls_embedding, last_hidden.transpose(1, 2))  # [B, 1, seq_len]
        attn_scores = attn_scores.squeeze(1)  # [B, seq_len]
        
        # Mask out padding tokens
        attn_scores = attn_scores.masked_fill(new_attention_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(1)  # [B, 1, seq_len]
        
        # Weighted sum of token embeddings
        context_vector = torch.bmm(attn_weights, last_hidden).squeeze(1)  # [B, hidden]
        
        # Combine CLS and attention-weighted context
        combined_embedding = (cls_embedding.squeeze(1) + context_vector) / 2
        
        # Project to output dimension
        global_emb = self.fc_global(combined_embedding)
        
        # Apply normalization but with a slight relaxation (0.9 instead of 1.0)
        # This preserves more of the original magnitude information
        global_emb_norm = torch.norm(global_emb, p=2, dim=1, keepdim=True)
        global_emb = global_emb / global_emb_norm.clamp(min=1e-8) * 0.9
        
        # Project each token representation for sequence embedding
        seq_emb = self.fc_seq(last_hidden)
        
        # Apply layer norm instead of L2 norm to sequence embeddings
        # This preserves relative magnitudes while stabilizing training
        seq_emb = torch.nn.functional.layer_norm(
            seq_emb, [seq_emb.size(-1)], eps=1e-5
        )
        
        return global_emb, seq_emb
