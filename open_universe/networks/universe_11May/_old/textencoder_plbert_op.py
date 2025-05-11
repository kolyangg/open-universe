import os
import yaml
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer
import sys

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
                # Use a placeholder token or set of tokens instead of empty
                token_ids = torch.LongTensor([0])  # Placeholder token ID
                batched.append(token_ids)
                max_len = max(max_len, 1)
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
        """
        phoneme_ids, attention_mask = self.tokenize(input_data)
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
        global_emb = self.fc_global(cls_embedding)
        # Project each token representation for sequence embedding
        seq_emb = self.fc_seq(outputs.last_hidden_state)
        return global_emb, seq_emb
