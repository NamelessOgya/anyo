import torch
from transformers import AutoTokenizer
from src.data.collators import BigRecCollator

from src.student.datamodule import SASRecDataModule
from src.data.collators import BigRecCollator
import torch
from transformers import AutoTokenizer
import os

def test_pipeline():
    print("Testing Pipeline with Real Data...")
    
    # 1. Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Setup DataModule
    data_dir = "data/ml-100k"
    if not os.path.exists(data_dir):
        print(f"Data dir {data_dir} not found!")
        return

    dm = SASRecDataModule(
        dataset_name="ml-100k",
        data_dir=data_dir,
        batch_size=2,
        max_seq_len=20, # Short for testing
        tokenizer=None, # We don't pass tokenizer here as we use custom collator
        num_workers=0,
        limit_data_rows=10 # Limit rows
    )
    dm.prepare_data()
    dm.setup()
    
    # 3. Setup Collator
    collator = BigRecCollator(
        tokenizer=tokenizer,
        item_id_to_name=dm.mapped_id_to_title,
        max_source_length=512,
        max_target_length=64,
        use_cot=False,
        max_history_items=20
    )
    
    # 4. Get a batch
    loader = torch.utils.data.DataLoader(
        dm.train_dataset,
        batch_size=2,
        collate_fn=collator,
        shuffle=False
    )
    
    batch = next(iter(loader))
    
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    next_items = batch["next_item"]
    
    print("\n--- Batch Info ---")
    for i in range(len(input_ids)):
        print(f"\n--- Sample {i} ---")
        
        # Decode Input
        decoded_input = tokenizer.decode(input_ids[i], skip_special_tokens=False)
        print(f"Full Input (Decoded): {decoded_input}")
        
        # Decode Labels
        label_ids = labels[i].clone()
        masked_indices = (label_ids == -100)
        label_ids[masked_indices] = tokenizer.pad_token_id
        decoded_labels = tokenizer.decode(label_ids, skip_special_tokens=False)
        
        # Get Target Name
        target_id = next_items[i].item()
        target_name = dm.mapped_id_to_title.get(target_id, f"Item_{target_id}")
        print(f"Target Name (ID {target_id}): {target_name}")
        
        # Check for Leakage in Prompt
        prompt_part = decoded_input.split("### Response:")[0]
        if target_name in prompt_part:
            print(f"WARNING: Target '{target_name}' found in Prompt! LEAKAGE DETECTED!")
        else:
            print(f"Target '{target_name}' NOT found in Prompt. (Good)")
            
        # Check Unmasked Labels
        unmasked_ids = labels[i][labels[i] != -100]
        decoded_unmasked = tokenizer.decode(unmasked_ids, skip_special_tokens=True)
        print(f"Unmasked Labels: '{decoded_unmasked}'")
        
        # Assertions
        if not decoded_unmasked.strip():
             raise AssertionError("Unmasked Labels are EMPTY! (Likely Over-masking)")
        
        # Check if target is in unmasked labels
        # Note: decoded_unmasked might have extra spaces or EOS
        if target_name not in decoded_unmasked:
             raise AssertionError(f"Target '{target_name}' NOT found in unmasked labels '{decoded_unmasked}'")
             
        # Check for leakage in prompt
        prompt_part = decoded_input.split("### Response:")[0]
        if target_name in prompt_part:
             raise AssertionError(f"Target '{target_name}' found in Prompt! LEAKAGE DETECTED!")

    print("\nTest Passed: No leakage detected and masking is correct.")

if __name__ == "__main__":
    test_pipeline()
