import os
import yaml
from collections import OrderedDict
import torch
import torch.nn as nn
from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer
import sys

import os
os.environ["TRUST_REMOTE_CODE"] = "True"

PLBERT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../_miipher/miipher2.0/plbert/'))
if PLBERT_PATH not in sys.path:
    sys.path.insert(0, PLBERT_PATH)

from phonemize import phonemize
from text_utils import TextCleaner

# (Make sure to have your global phonemizer defined elsewhere, for example:)
from phonemizer.backend import EspeakBackend

global_phonemizer = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)

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
        # plbert_root = "plbert/"
        plbert_root = PLBERT_PATH
        log_dir = os.path.join(plbert_root, "Checkpoint")
        # config_path = os.path.join(log_dir, "config.yml")
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
            name = k[7:]  # remove `module.`
            if name.startswith('encoder.'):
                name = name[8:]  # remove `encoder.`
                new_state_dict[name] = v

        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in self.plbert.state_dict()}
        self.plbert.load_state_dict(filtered_state_dict, strict=False)

        # self.plbert.load_state_dict(new_state_dict)
        
        # Instantiate the tokenizer from PL-BERT config
        self.tokenizer = TransfoXLTokenizer.from_pretrained(plbert_config['dataset_params']['tokenizer'])
        self.text_cleaner = TextCleaner()
        
        # Projection layers for global and sequence outputs.
        self.fc_global = nn.Linear(self.plbert.config.hidden_size, hidden_dim)
        if seq_dim is None:
            seq_dim = hidden_dim
        self.fc_seq = nn.Linear(self.plbert.config.hidden_size, seq_dim)
        
        if freeze_plbert:
            for param in self.plbert.parameters():
                param.requires_grad = False

    def tokenize(self, sents):
        batched = []
        max_len = 0
        for sent in sents:
            # Phonemize the sentence using the global phonemizer
            phonemes = phonemize(sent, global_phonemizer, self.tokenizer)['phonemes']
            pretext = ' '.join(phonemes)
            cleaned = self.text_cleaner(pretext)
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
        
        outputs = self.plbert(phoneme_ids, attention_mask=attention_mask)
        # Use first token (CLS) for global embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        global_emb = self.fc_global(cls_embedding)
        # Project each token representation for sequence embedding
        seq_emb = self.fc_seq(outputs.last_hidden_state)
        return global_emb, seq_emb
