import torch
from src.student.datamodule import TeacherTrainCollater
from transformers import AutoTokenizer
import pytest

def test_collator_leakage():
    # Mock Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize Collater
    collater = TeacherTrainCollater(
        tokenizer=tokenizer,
        max_seq_len=10,
        padding_item_id=0,
        id_to_name={1: "Item 1", 2: "Item 2"}
    )
    
    # Create a dummy batch
    batch = [{
        "seq_ids": [1],
        "next_item_id": 2,
        "candidates": [3, 4],
        "has_teacher_target": False
    }]
    
    # Run Collater
    output = collater(batch)
    input_ids = output["input_ids"][0]
    
    # Check if Target (Item 2) is in input_ids
    # Item 2 token ID = vocab_offset + 2
    target_token_id = collater.vocab_offset + 2
    
    print(f"Input IDs: {input_ids}")
    print(f"Target Token ID: {target_token_id}")
    
    # It SHOULD NOT be in input_ids for correct training (predicting next item)
    # But currently we suspect it IS there.
    if target_token_id in input_ids:
        print("LEAKAGE DETECTED: Target token found in input_ids!")
    else:
        print("NO LEAKAGE: Target token not found in input_ids.")
        
    # Also check the last token
    last_token = input_ids[output["attention_mask"][0].sum() - 1]
    print(f"Last Token: {last_token}")
    
    assert target_token_id not in input_ids, "Target token should NOT be in input_ids"
    assert last_token != target_token_id, "Last token should NOT be target"
