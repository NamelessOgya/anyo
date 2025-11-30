
import torch
from transformers import AutoTokenizer

def reproduce_ilora_bug():
    print("--- Reproducing iLoRA Padding Bug ---")
    
    # 1. Setup Tokenizer with Right Padding (as in run_teacher.py)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        print("gpt2 not found, skipping")
        return

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")
    
    # 2. Create Input
    texts = ["short", "a very long sentence to ensure different lengths"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    print(f"\nInput IDs:\n{input_ids}")
    print(f"Attention Mask:\n{attention_mask}")
    
    # 3. Simulate Logic in iLoRA (Buggy)
    # last_hidden_state[:, -1, :]
    # We just check the token ID at -1
    buggy_indices = -1
    buggy_tokens = input_ids[:, buggy_indices]
    
    print("\n[Buggy Logic] Tokens at index -1:")
    for i, token_id in enumerate(buggy_tokens):
        print(f"Text {i}: {token_id} ('{tokenizer.decode([token_id])}')")
        if token_id == tokenizer.pad_token_id:
            print(f"  -> ERROR: This is a PAD token!")
        else:
            print(f"  -> OK (but might be wrong if text was padded)")

    # 4. Correct Logic for Right Padding
    # attention_mask.sum(1) - 1
    correct_indices = attention_mask.sum(1) - 1
    correct_tokens = input_ids[torch.arange(len(texts)), correct_indices]
    
    print("\n[Correct Logic] Tokens at attention_mask.sum(1) - 1:")
    for i, token_id in enumerate(correct_tokens):
        print(f"Text {i}: {token_id} ('{tokenizer.decode([token_id])}')")
        if token_id == tokenizer.pad_token_id:
            print(f"  -> ERROR: This is a PAD token!")
        else:
            print(f"  -> SUCCESS: This is the last real token.")

if __name__ == "__main__":
    reproduce_ilora_bug()
