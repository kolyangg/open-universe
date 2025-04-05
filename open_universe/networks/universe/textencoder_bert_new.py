import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim, seq_dim=None, freeze_bert=True):
        """
        Args:
            hidden_dim (int): Dimension of the global output embedding.
            seq_dim (int, optional): Dimension of the sequence embeddings for cross-attention.
                                     Defaults to hidden_dim if not provided.
            freeze_bert (bool): Whether to freeze BERT weights during training.
        """
        super(TextEncoder, self).__init__()
        
        # Initialize standard BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Only need sequence projection since we're not using FiLM
        if seq_dim is None:
            seq_dim = hidden_dim
        self.fc_seq = nn.Linear(self.bert.config.hidden_size, seq_dim)
        
        # Initialize projection layers properly
        nn.init.xavier_uniform_(self.fc_seq.weight)
        nn.init.zeros_(self.fc_seq.bias)
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.max_length = 128  # Max sequence length to process
        
    def forward(self, input_data):
        """
        Args:
            input_data (list of str): List of sentences.
        Returns:
            _, seq_emb: Tuple with empty placeholder for global embedding and sequence embeddings
        """
        # Skip processing for empty batch
        if len(input_data) == 0:
            device = next(self.bert.parameters()).device
            return (
                None,  # We don't need global embeddings when not using FiLM
                torch.zeros((0, 0, self.fc_seq.out_features), device=device)
            )
        
        # Tokenize the input texts
        encoded_inputs = self.tokenizer(
            input_data,
            padding='longest',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        device = next(self.bert.parameters()).device
        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)
        
        # Log tokenization details for debugging
        print(f"[DEBUG] Sample input text: '{input_data[0]}'")
        print(f"[DEBUG] Token IDs shape: {input_ids.shape}")
        print(f"[DEBUG] Active tokens: {attention_mask.sum(dim=1).tolist()}")
        
        # Process through BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        # Project each token representation for sequence embedding
        seq_emb = self.fc_seq(outputs.last_hidden_state)
        
        # Return None for global embedding since we don't use FiLM
        return None, seq_emb