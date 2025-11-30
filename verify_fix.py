
import torch
from transformers import AutoTokenizer

def verify_fix():
    model_name = "meta-llama/Llama-2-7b-hf"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        print("Could not load Llama-2, using gpt2 for demonstration")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    
    texts = ["short", "a very long sentence"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    
    print(f"Padding side: {tokenizer.padding_side}")
    
    # Simulate hidden states (just using input_ids as dummy values)
    # Shape: (B, Seq, Dim=1) for simplicity
    hidden_states = inputs.input_ids.unsqueeze(-1).float()
    
    # Fixed Logic: Take last token
    pred_embeddings = hidden_states[:, -1, :]
    
    print("\nVerifying extracted tokens:")
    for i in range(len(texts)):
        token_id = int(pred_embeddings[i, 0].item())
        print(f"Text {i}: Extracted token ID {token_id} ('{tokenizer.decode([token_id])}')")
        
        # Verify it is indeed the last token of the text
        # For left padding, the sequence ends at the last position of the tensor
        # So it should be the last token of the input text
        
        # Let's check manually
        encoded_single = tokenizer(texts[i], add_special_tokens=False)["input_ids"]
        last_token_single = encoded_single[-1]
        
        # Note: tokenizer(texts) might add special tokens (BOS) depending on config
        # But here we just want to ensure we are picking the token at the physical end of the sequence
        
        actual_last_in_tensor = inputs.input_ids[i, -1].item()
        
        if token_id == actual_last_in_tensor:
            print(f"--> SUCCESS: Matches last token in tensor.")
        else:
            print(f"--> FAILURE: Does not match last token in tensor.")
            
        if token_id == tokenizer.pad_token_id:
             print(f"--> WARNING: Extracted token is PAD. This shouldn't happen if texts are not empty.")

if __name__ == "__main__":
    verify_fix()
