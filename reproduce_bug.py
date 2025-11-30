
import torch
from transformers import AutoTokenizer

def test_padding_logic():
    model_name = "meta-llama/Llama-2-7b-hf"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        # Fallback to a smaller model if Llama-2 requires auth or is large
        print("Could not load Llama-2, using gpt2 for demonstration (setting pad token manually)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding to left (as in BigRecModel)
    tokenizer.padding_side = "left"
    
    texts = ["short", "a very long sentence"]
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    
    print(f"Padding side: {tokenizer.padding_side}")
    print(f"Input IDs:\n{inputs.input_ids}")
    print(f"Attention Mask:\n{inputs.attention_mask}")
    
    # Buggy logic from BigRecModel
    last_token_idx_buggy = inputs.attention_mask.sum(1) - 1
    print(f"\nBuggy Index (sum-1): {last_token_idx_buggy}")
    
    # Check what token is at the buggy index
    for i in range(len(texts)):
        idx = last_token_idx_buggy[i].item()
        token_id = inputs.input_ids[i, idx].item()
        print(f"Text {i}: Token at buggy index {idx} is {token_id} ('{tokenizer.decode([token_id])}')")
        
        # Correct index should be -1 (last token)
        correct_idx = -1
        token_id_correct = inputs.input_ids[i, correct_idx].item()
        print(f"Text {i}: Token at correct index {correct_idx} is {token_id_correct} ('{tokenizer.decode([token_id_correct])}')")
        
        if token_id == tokenizer.pad_token_id:
            print(f"--> ERROR: Buggy index points to PAD token!")
        elif idx != inputs.input_ids.shape[1] - 1:
             print(f"--> ERROR: Buggy index {idx} does not point to the end of the sequence (len {inputs.input_ids.shape[1]})")

if __name__ == "__main__":
    test_padding_logic()
