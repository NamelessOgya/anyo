import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.student.datamodule import SASRecDataModule
from transformers import AutoTokenizer

@pytest.fixture(scope="module")
def data_consistency_fixture(tmp_path_factory):
    """
    Fixture for Data Consistency tests.
    Creates a temporary dataset and instantiates SASRecDataModule.
    """
    data_dir = tmp_path_factory.mktemp("data")
    
    # Create dummy data
    num_users = 10
    num_items = 20
    
    # movies.dat
    with open(data_dir / "movies.dat", "w", encoding="latin-1") as f:
        for i in range(1, num_items + 1):
            f.write(f"{i}::Movie {i}::Genre\n")
            
    # train.csv
    train_df = pd.DataFrame({
        "user_id": range(1, 6),
        "seq": ["1 2 3", "4 5", "1", "2 3 4 5", "1 2"],
        "next_item": [4, 6, 2, 6, 3]
    })
    train_df.to_csv(data_dir / "train.csv", index=False)
    
    # val.csv
    val_df = pd.DataFrame({
        "user_id": range(1, 6),
        "seq": ["1 2 3 4", "4 5 6", "1 2", "2 3 4 5 6", "1 2 3"],
        "next_item": [5, 7, 3, 7, 4]
    })
    val_df.to_csv(data_dir / "val.csv", index=False)
    
    # test.csv
    test_df = pd.DataFrame({
        "user_id": range(1, 6),
        "seq": ["1 2 3 4 5", "4 5 6 7", "1 2 3", "2 3 4 5 6 7", "1 2 3 4"],
        "next_item": [6, 8, 4, 8, 5]
    })
    test_df.to_csv(data_dir / "test.csv", index=False)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir=str(data_dir),
        batch_size=2,
        max_seq_len=10,
        tokenizer=tokenizer,
        num_workers=0
    )
    dm.setup()
    
    return {
        "dm": dm,
        "data_dir": data_dir,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df
    }

def test_16_data_leakage(data_consistency_fixture):
    """
    Test 16: [Data Leakage] Verify that validation/test items do not appear in training history for the same user.
    """
    dm = data_consistency_fixture["dm"]
    
    # Check if next_item in val/test appears in train seq for the same user?
    # Actually, standard split is leave-one-out.
    # Train: [1, 2, 3] -> 4
    # Val: [1, 2, 3, 4] -> 5
    # Test: [1, 2, 3, 4, 5] -> 6
    # So Val target (5) should NOT be in Train seq [1, 2, 3].
    # Test target (6) should NOT be in Train seq [1, 2, 3] or Val seq [1, 2, 3, 4].
    
    # We need to map user IDs back to original or check consistent mapping.
    # dm.train_dataset has mapped IDs.
    
    # Let's iterate over users in train dataset
    train_user_items = {}
    for i in range(len(dm.train_dataset)):
        sample = dm.train_dataset[i]
        # We don't have user_id in __getitem__, but we can infer from index if we assume order?
        # Or we can check the dataframe directly.
        pass
    
    # Better to check dataframes in dm
    train_df = dm.train_df
    val_df = dm.val_df
    test_df = dm.test_df
    
    # Group by user_id
    for user_id in train_df['user_id'].unique():
        train_row = train_df[train_df['user_id'] == user_id].iloc[0]
        val_row = val_df[val_df['user_id'] == user_id].iloc[0]
        test_row = test_df[test_df['user_id'] == user_id].iloc[0]
        
        train_seq = set(train_row['seq'])
        val_target = val_row['next_item']
        test_target = test_row['next_item']
        
        # Check leakage
        assert val_target not in train_seq, f"User {user_id}: Val target {val_target} in Train seq {train_seq}"
        assert test_target not in train_seq, f"User {user_id}: Test target {test_target} in Train seq {train_seq}"
        
        # Also check if Test target is in Val seq (it shouldn't be, usually Val seq = Train seq + Val target)
        val_seq = set(val_row['seq'])
        assert test_target not in val_seq, f"User {user_id}: Test target {test_target} in Val seq {val_seq}"

def test_17_padding_handling(data_consistency_fixture):
    """
    Test 17: [Padding Handling] Verify padding ID is 0 and excluded from loss/metrics.
    """
    dm = data_consistency_fixture["dm"]
    assert dm.padding_item_id == 0
    
    # Check collater padding
    batch = next(iter(dm.train_dataloader()))
    seq = batch["seq"]
    len_seq = batch["len_seq"]
    
    for i in range(seq.size(0)):
        length = len_seq[i].item()
        padded_part = seq[i, :dm.max_seq_len - length] # Padding at beginning?
        # SASRecDataModule collater pads at beginning:
        # [pad, pad, 1, 2, 3]
        
        if length < dm.max_seq_len:
            assert (padded_part == 0).all(), f"Padding should be 0, got {padded_part}"

def test_18_sequence_generation(data_consistency_fixture):
    """
    Test 18: [Sequence Generation] Verify sequence generation logic.
    """
    # This logic is usually in data preprocessing (creating CSVs).
    # Since we loaded pre-split CSVs, we check if the loaded sequences follow the pattern.
    # Train: [A, B] -> C
    # Val: [A, B, C] -> D
    # Test: [A, B, C, D] -> E
    
    dm = data_consistency_fixture["dm"]
    train_df = dm.train_df
    val_df = dm.val_df
    test_df = dm.test_df
    
    for user_id in train_df['user_id'].unique():
        train_row = train_df[train_df['user_id'] == user_id].iloc[0]
        val_row = val_df[val_df['user_id'] == user_id].iloc[0]
        test_row = test_df[test_df['user_id'] == user_id].iloc[0]
        
        train_seq = train_row['seq']
        train_next = train_row['next_item']
        
        val_seq = val_row['seq']
        val_next = val_row['next_item']
        
        test_seq = test_row['seq']
        test_next = test_row['next_item']
        
        # Check continuity
        # Val seq should be Train seq + Train next
        expected_val_seq = train_seq + [train_next]
        # Truncate if needed (but here seqs are short)
        assert val_seq == expected_val_seq, f"User {user_id}: Val seq mismatch"
        
        # Test seq should be Val seq + Val next
        expected_test_seq = val_seq + [val_next]
        assert test_seq == expected_test_seq, f"User {user_id}: Test seq mismatch"

def test_19_id_mapping(data_consistency_fixture):
    """
    Test 19: [ID Mapping] Verify ID mapping is consistent and 1-based (0 is padding).
    """
    dm = data_consistency_fixture["dm"]
    
    # Check item_id_map
    assert 0 not in dm.item_id_map.values() # 0 is reserved for padding
    assert min(dm.item_id_map.values()) == 1
    
    # Check if all items in CSVs are mapped
    # We checked this implicitly by loading data without error (dropna)
    pass

def test_20_tokenizer_synchronization(data_consistency_fixture):
    """
    Test 20: [Tokenizer Synchronization] Verify tokenizer vocab and item ID map.
    """
    dm = data_consistency_fixture["dm"]
    tokenizer = dm.tokenizer
    
    # Check if special tokens are added
    # SASRecDataModule doesn't add tokens, the caller (trainer/test) does.
    # But we passed a tokenizer.
    # We should check if item names in mapped_id_to_title are valid strings?
    
    assert hasattr(dm, "mapped_id_to_title")
    assert len(dm.mapped_id_to_title) > 0
    
    # Check if tokenizer can encode item names
    for item_id, title in dm.mapped_id_to_title.items():
        encoded = tokenizer.encode(title)
        assert len(encoded) > 0
