import os
import sys
import torch
import numpy as np
from collections import OrderedDict
import yaml
import matplotlib.pyplot as plt
from transformers import AlbertConfig, AlbertModel, TransfoXLTokenizer

# Path to your TextEncoder
import sys
#sys.path.append('/path/to/your/project')  # Adjust this path to where your TextEncoder is located
from textencoder_plbert_op import TextEncoder  # Import your TextEncoder class

def test_plbert_encoder():
    """Test the PL-BERT encoder functionality with detailed debugging"""
    print("="*50)
    print("TESTING PL-BERT TEXT ENCODER")
    print("="*50)
    
    # 1. Initialize the encoder
    print("\n[STEP 1] Initializing TextEncoder...")
    try:
        encoder = TextEncoder(hidden_dim=256, seq_dim=256, freeze_plbert=True)
        print("[SUCCESS] TextEncoder initialized")
        
        # Check model parameters
        plbert_params = sum(p.numel() for p in encoder.plbert.parameters())
        proj_params = sum(p.numel() for p in encoder.fc_global.parameters()) + \
                      sum(p.numel() for p in encoder.fc_seq.parameters())
        
        print(f"PL-BERT parameters: {plbert_params:,}")
        print(f"Projection layers parameters: {proj_params:,}")
        print(f"Total parameters: {plbert_params + proj_params:,}")
        
        # Check which parameters are trainable
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize TextEncoder: {e}")
        raise e
    
    # 2. Test tokenization
    print("\n[STEP 2] Testing tokenization...")
    test_sentences = [
        "Hello world, this is a test.",
        "Speech enhancement is fascinating.",
        "I wonder how this will be phonemized?",
        ""  # Empty string to test edge case
    ]
    
    try:
        # Test phonemization
        print("Testing phonemization...")
        for i, sent in enumerate(test_sentences):
            if sent:  # Skip empty string
                phonemized = encoder.phonemizer(sent)
                print(f"  Sentence {i+1}: '{sent}'")
                print(f"  Phonemized: '{phonemized}'")
                print()
        
        # Test full tokenization
        print("Testing tokenization...")
        phoneme_ids, attention_mask = encoder.tokenize(test_sentences)
        print(f"Phoneme IDs shape: {phoneme_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        
        # Show token statistics
        token_counts = attention_mask.sum(dim=1).tolist()
        print(f"Token counts per sentence: {token_counts}")
        
    except Exception as e:
        print(f"[ERROR] Failed in tokenization step: {e}")
        raise e
        
    # 3. Test forward pass
    print("\n[STEP 3] Testing forward pass...")
    try:
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        print(f"Using device: {device}")
        
        # Process the sentences
        with torch.no_grad():
            global_emb, seq_emb = encoder(test_sentences)
        
        print(f"Global embeddings shape: {global_emb.shape}")
        print(f"Sequence embeddings shape: {seq_emb.shape}")
        
        # Check embedding statistics
        print("\nGlobal embedding statistics:")
        print(f"  Mean: {global_emb.mean().item():.4f}")
        print(f"  Std: {global_emb.std().item():.4f}")
        print(f"  Min: {global_emb.min().item():.4f}")
        print(f"  Max: {global_emb.max().item():.4f}")
        
        print("\nSequence embedding statistics:")
        print(f"  Mean: {seq_emb.mean().item():.4f}")
        print(f"  Std: {seq_emb.std().item():.4f}")
        print(f"  Min: {seq_emb.min().item():.4f}")
        print(f"  Max: {seq_emb.max().item():.4f}")
        
        # Visualize embeddings
        plt.figure(figsize=(12, 6))
        
        # Plot global embeddings
        plt.subplot(1, 2, 1)
        for i in range(min(3, len(test_sentences))):
            if test_sentences[i]:  # Skip empty string
                plt.plot(global_emb[i].cpu().numpy(), label=f"Sentence {i+1}")
        plt.title("Global Embeddings")
        plt.legend()
        
        # Plot first token sequence embeddings
        plt.subplot(1, 2, 2)
        for i in range(min(3, len(test_sentences))):
            if test_sentences[i]:  # Skip empty string
                plt.plot(seq_emb[i, 0].cpu().numpy(), label=f"Sentence {i+1} (first token)")
        plt.title("First Token Sequence Embeddings")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("plbert_embeddings.png")
        print("\nEmbedding visualization saved to 'plbert_embeddings.png'")
        
    except Exception as e:
        print(f"[ERROR] Failed in forward pass: {e}")
        raise e
    
    # 4. Test with a larger batch of sentences
    print("\n[STEP 4] Testing with larger batch...")
    try:
        # Generate more test sentences
        more_sentences = [f"This is test sentence number {i+1}." for i in range(10)]
        
        # Process the sentences
        with torch.no_grad():
            global_emb, seq_emb = encoder(more_sentences)
        
        print(f"Successfully processed batch of {len(more_sentences)} sentences")
        print(f"Global embeddings shape: {global_emb.shape}")
        print(f"Sequence embeddings shape: {seq_emb.shape}")
        
    except Exception as e:
        print(f"[ERROR] Failed with larger batch: {e}")
        raise e
        
    # 5. Test cache functionality
    print("\n[STEP 5] Testing cache functionality...")
    try:
        # Record initial cache size
        initial_cache_size = len(encoder.phoneme_cache)
        print(f"Initial cache size: {initial_cache_size}")
        
        # Process same sentences again
        with torch.no_grad():
            global_emb, seq_emb = encoder(test_sentences)
            
        # Check cache size after
        final_cache_size = len(encoder.phoneme_cache)
        print(f"Final cache size: {final_cache_size}")
        print(f"New items in cache: {final_cache_size - initial_cache_size}")
        
        # Check cache contents
        print("\nCache contents:")
        for i, (k, v) in enumerate(list(encoder.phoneme_cache.items())[:3]):  # Show first 3
            print(f"  '{k[:30]}...' -> '{v[:30]}...'")
        
    except Exception as e:
        print(f"[ERROR] Failed testing cache: {e}")
        raise e
        
    print("\n[COMPLETE] All tests completed successfully!")
    print("="*50)

if __name__ == "__main__":
    test_plbert_encoder()