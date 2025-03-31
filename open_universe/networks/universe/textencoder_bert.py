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
        # A fully connected layer to project BERT's hidden size to your desired hidden_dim
        self.fc = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        
        # Optionally freeze BERT parameters to stabilize early training
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Token IDs tensor of shape (batch_size, seq_length).
            attention_mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length) 
                indicating which tokens should be attended to.
                
        Returns:
            torch.Tensor: Encoded text features of shape (batch_size, hidden_dim).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation (first token) as the summary embedding.
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size)
        return self.fc(cls_embedding)
