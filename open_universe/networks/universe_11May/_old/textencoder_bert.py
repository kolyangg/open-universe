from transformers import DistilBertModel, DistilBertTokenizer
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim, freeze_bert=True):
        """
        Args:
            hidden_dim (int): Dimension of the output features.
            freeze_bert (bool): Whether to freeze the DistilBERT weights during training.
        """
        super(TextEncoder, self).__init__()
        # Load the pre-trained DistilBERT model
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Load the tokenizer as well
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # A fully connected layer to project BERT's hidden size to your desired hidden_dim
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
        # Optionally freeze BERT parameters to stabilize early training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_data, attention_mask=None):
        """
        Args:
            input_data: Either a string, a list of strings, or a tensor of token IDs.
            attention_mask (torch.Tensor, optional): Mask tensor if token IDs are already provided.
        Returns:
            torch.Tensor: Encoded text features of shape (batch_size, hidden_dim).
        """
        # If input_data is a string or a list of strings, tokenize it.
        if isinstance(input_data, str) or (isinstance(input_data, list) and isinstance(input_data[0], str)):
            encoded = self.tokenizer(input_data, return_tensors="pt", padding=True, truncation=True)
            # Move token tensors to the same device as BERT.
            input_ids = encoded["input_ids"].to(next(self.bert.parameters()).device)
            attention_mask = encoded["attention_mask"].to(next(self.bert.parameters()).device)
        else:
            # Assume input_data is already a tensor of token IDs.
            input_ids = input_data
            # If attention_mask was not provided, create one (assuming non-padding tokens are non-zero).
            if attention_mask is None:
                attention_mask = (input_ids != 0).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # # debug print
        # print(f"[DEBUG] BERT output shape: {outputs.last_hidden_state.shape}")
        # print(f"[DEBUG] Attention mask shape: {attention_mask.shape}")
        # print(f"[DEBUG] Input IDs shape: {input_ids.shape}")
        # Use the [CLS] token representation (first token) as the summary embedding.
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        return self.fc(cls_embedding)  # (batch_size, hidden_dim)
