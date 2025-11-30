
import torch
import pytest
from transformers import AutoTokenizer

def test_left_padding_last_token_extraction():
    """
    Verifies that we can correctly extract the last token from a left-padded sequence.
    This prevents the regression where we used logic assuming right-padding.
    """
    # Use a small model or just a tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        pytest.skip("gpt2 tokenizer not available")
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    texts = ["short", "a very long sentence to ensure different lengths"]
    
    # Tokenize with padding
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Logic used in BigRecModel (Corrected)
    # With left padding, the last token is always at index -1
    extracted_token_ids = input_ids[:, -1]
    
    # Verify
    for i, text in enumerate(texts):
        # Get the actual last token by tokenizing individually without padding
        single_input = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        expected_last_token = single_input.input_ids[0, -1]
        
        assert extracted_token_ids[i] == expected_last_token, \
            f"Failed for text '{text}': Expected {expected_last_token}, got {extracted_token_ids[i]}"

def test_right_padding_last_token_extraction():
    """
    Verifies the logic for right padding, just for completeness and comparison.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        pytest.skip("gpt2 tokenizer not available")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    texts = ["short", "a very long sentence to ensure different lengths"]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Logic for right padding: last real token is at sum(mask) - 1
    last_token_indices = attention_mask.sum(1) - 1
    extracted_token_ids = input_ids[torch.arange(len(texts)), last_token_indices]
    
    # Verify
    for i, text in enumerate(texts):
        single_input = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        expected_last_token = single_input.input_ids[0, -1]
        
        assert extracted_token_ids[i] == expected_last_token, \
            f"Failed for text '{text}': Expected {expected_last_token}, got {extracted_token_ids[i]}"
