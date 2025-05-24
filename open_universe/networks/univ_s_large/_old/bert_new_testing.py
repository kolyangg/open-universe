#!/usr/bin/env python
# Simple test script for the BERT Text Encoder

import torch
from textencoder_bert_new import TextEncoder  # Assuming your code is in text_encoder.py

def main():
    print("Testing BERT Text Encoder")
    print("-" * 50)
    
    # Create the encoder with default settings
    encoder = TextEncoder(hidden_dim=256, freeze_bert=True)
    print(f"Initialized TextEncoder with hidden_dim=256")
    
    # Test with simple examples
    examples = [
        ["Hello world, this is a simple test."],
        ["Speech enhancement is improved with text conditioning."],
        ["This is an example of a longer sentence that talks about audio processing and how it might benefit from contextual information contained in transcripts."],
        [""],  # Empty string test
        ["Hello", "Multiple sentences", "In a batch"],  # Batch processing test
    ]
    
    # Process each example
    for i, texts in enumerate(examples):
        print(f"\nExample {i+1}: {texts}")
        print("-" * 30)
        
        # Forward pass
        _, sequence_embeddings = encoder(texts)
        
        # Print stats about embeddings
        if sequence_embeddings is not None:
            print(f"Sequence embedding shape: {sequence_embeddings.shape}")
            print(f"Embedding stats - Min: {sequence_embeddings.min().item():.4f}, Max: {sequence_embeddings.max().item():.4f}")
            print(f"Embedding norm (mean across sequence): {sequence_embeddings.norm(dim=2).mean().item():.4f}")
            
            # Print attention visualization (simplified)
            if len(texts) == 1 and texts[0]:  # For single non-empty text
                # Get the original tokens
                tokens = encoder.tokenizer.tokenize(texts[0])
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                tokens = tokens[:sequence_embeddings.shape[1]]  # Truncate if needed
                
                # Calculate token importance (using L2 norm of each token's embedding)
                token_norms = sequence_embeddings[0, :len(tokens)].norm(dim=1)
                
                # Print top 5 tokens by importance
                if len(tokens) > 0:
                    norm_values, indices = torch.topk(token_norms, min(5, len(tokens)))
                    print("\nTop 5 tokens by embedding norm:")
                    for idx, (token_idx, norm_val) in enumerate(zip(indices.tolist(), norm_values.tolist())):
                        if token_idx < len(tokens):
                            print(f"  {idx+1}. '{tokens[token_idx]}' (position {token_idx}): {norm_val:.4f}")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()