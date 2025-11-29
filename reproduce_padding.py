
import sys
import os
import torch
from transformers import AutoTokenizer
from src.student.datamodule import SASRecDataModule

def main():
    # Mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[HistoryEmb]', '[CansEmb]', '[ItemEmb]']})
    tokenizer.pad_token = tokenizer.eos_token

    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="data/ml-1m",
        batch_size=4,
        max_seq_len=50,
        num_workers=0,
        limit_data_rows=100,
        tokenizer=tokenizer
    )
    dm.prepare_data()
    dm.setup()

    print("\n--- Checking Train Dataloader Batch Shapes ---")
    dataloader = dm.train_dataloader()
    
    max_len_seen = 0
    min_len_seen = 10000
    
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids']
        shape = input_ids.shape
        print(f"Batch {i}: input_ids shape = {shape}")
        
        if shape[1] > max_len_seen:
            max_len_seen = shape[1]
        if shape[1] < min_len_seen:
            min_len_seen = shape[1]
            
        if i >= 5: # Check a few batches
            break
            
    print(f"\nMax length seen: {max_len_seen}")
    print(f"Min length seen: {min_len_seen}")
    
    if max_len_seen < 512 and min_len_seen < 512:
        print("\nSUCCESS: Batch lengths are dynamic and less than 512.")
    elif max_len_seen == 512 and min_len_seen == 512:
        print("\nWARNING: Batch lengths seem fixed at 512 (or data naturally reaches 512).")
    else:
        print("\nObservation: Batch lengths vary.")

if __name__ == "__main__":
    main()
