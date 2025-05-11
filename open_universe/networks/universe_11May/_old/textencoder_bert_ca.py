from transformers import DistilBertModel, DistilBertTokenizer
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim, seq_dim=None, freeze_bert=True):
        """
        Args:
            hidden_dim (int): Dimension of the global output embedding (for FiLM).
            seq_dim (int, optional): Dimension of the sequence embeddings (for cross-attention).
                                     Defaults to hidden_dim if not provided.
            freeze_bert (bool): Whether to freeze the DistilBERT weights during training.
        """
        super(TextEncoder, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.fc_global = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        # Use seq_dim if provided, otherwise default to hidden_dim
        if seq_dim is None:
            seq_dim = hidden_dim
        self.fc_seq = nn.Linear(self.bert.config.hidden_size, seq_dim)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_data, attention_mask=None):
        # If input_data is a string or list of strings, tokenize it.
        if isinstance(input_data, str) or (isinstance(input_data, list) and isinstance(input_data[0], str)):
            encoded = self.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded["input_ids"].to(next(self.bert.parameters()).device)
            attention_mask = encoded["attention_mask"].to(next(self.bert.parameters()).device)
        else:
            input_ids = input_data
            if attention_mask is None:
                attention_mask = (input_ids != 0).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Global embedding: use the [CLS] token (first token) and project it.
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, bert_hidden_dim)
        global_emb = self.fc_global(cls_embedding)            # (batch_size, hidden_dim)
        # Sequence embedding: project each token's embedding.
        seq_emb = self.fc_seq(outputs.last_hidden_state)       # (batch_size, seq_length, seq_dim)
        return global_emb, seq_emb
