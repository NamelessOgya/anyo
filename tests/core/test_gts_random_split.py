import pytest
import pandas as pd
import numpy as np
from src.core.preprocess_data import process_and_split

def test_gts_random_split_logic():
    # 1. Setup Synthetic Data
    # Create 2 users.
    # User 1: 10 items. Timestamps 100, 200, ..., 1000.
    # User 2: 10 items. Timestamps 100, 200, ..., 1000.
    # Split ratio 0.8 -> Split time should be around 800 (depending on exact quantile logic).
    # Let's make it explicit.
    
    users = [1] * 10 + [2] * 10
    items = list(range(1, 11)) + list(range(11, 21))
    timestamps = list(range(100, 1100, 100)) + list(range(100, 1100, 100))
    
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'timestamp': timestamps
    })
    
    # Calculate expected split time manually to verify
    # Global sort by timestamp: 20 items. 
    # 100, 100, 200, 200, ..., 1000, 1000.
    # Split ratio 0.8 -> Index int(20 * 0.8) = 16.
    # Sorted timestamps:
    # Indices 0-1: 100
    # ...
    # Indices 14-15: 800
    # Index 16: 900.
    # So split_time should be 900.
    # Items with timestamp <= 900 are TRAIN period.
    # Items with timestamp > 900 are HOLDOUT (Test) period.
    # Wait, my implementation:
    # split_idx = int(len(df_sorted) * split_ratio)
    # split_time = df_sorted.iloc[split_idx]['timestamp']
    # If split_idx is 16, split_time is 900.
    # train_indices = [t <= split_time] -> timestamps <= 900.
    # holdout_indices = [t > split_time] -> timestamps > 900 (i.e., 1000).
    
    # Let's adjust timestamps to make it more interesting.
    # User 1: 100..1000
    # User 2: 100..1000
    # Let's make split_ratio such that we have a clear cut.
    # If I use 0.8, split index is 16.
    # Sorted timestamps: [100, 100, ..., 800, 800, 900, 900, 1000, 1000]
    # Index 16 is 900 (one of the 900s).
    # So split_time = 900.
    # Train period: <= 900. (100..900) -> 9 items per user.
    # Holdout period: > 900. (1000) -> 1 item per user.
    
    # This might be too few for holdout (only 1 item).
    # Let's set split_ratio to 0.6.
    # Index = 20 * 0.6 = 12.
    # Sorted: ... 600, 600 (idx 10, 11), 700, 700 (idx 12, 13).
    # split_time = 700.
    # Train period: <= 700. (100..700) -> 7 items.
    # Holdout period: > 700. (800, 900, 1000) -> 3 items.
    
    split_ratio = 0.6
    
    # Run process_and_split
    # We need to mock logging or just ignore it.
    train_df, val_df, test_df = process_and_split(df, min_seq_len=3, split_method='gts-random', split_ratio=split_ratio)
    
    # Re-calculate split time for verification
    df_sorted = df.sort_values('timestamp')
    split_idx = int(len(df_sorted) * split_ratio)
    split_time = df_sorted.iloc[split_idx]['timestamp']
    print(f"Split time: {split_time}")
    
    # Helper to parse seq string back to list
    def parse_seq(seq_str):
        return [int(x) for x in seq_str.split()]
    
    # --- Verify Condition 1: Validation data is the last training item ---
    # "全てのvalidation dataがtrain期間の最新データと一致していること"
    print("Verifying Condition 1...")
    for _, row in val_df.iterrows():
        user_id = row['user_id']
        val_item = row['next_item']
        
        # Get user's full history
        user_hist = df[df['user_id'] == user_id].sort_values('timestamp')
        
        # Get items in train period
        train_items = user_hist[user_hist['timestamp'] <= split_time]
        
        # The validation item should be the LAST item of the train_items
        expected_val_item = train_items.iloc[-1]['item_id']
        
        assert val_item == expected_val_item, f"User {user_id}: Val item {val_item} != Expected {expected_val_item}"
        
        # Also verify val sequence is everything before it
        val_seq = parse_seq(row['seq'])
        expected_val_seq = train_items.iloc[:-1]['item_id'].tolist()
        assert val_seq == expected_val_seq, f"User {user_id}: Val seq mismatch"

    # --- Verify Condition 2: Test data is after split time ---
    # "全てのtest dataがtest期間の後に含まれるデータであること"
    print("Verifying Condition 2...")
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        test_item = row['next_item']
        
        # Get timestamp of test item
        # Note: item_id is unique in this synthetic data so we can look it up directly
        item_ts = df[df['item_id'] == test_item]['timestamp'].values[0]
        
        assert item_ts > split_time, f"User {user_id}: Test item {test_item} timestamp {item_ts} <= split_time {split_time}"

    # --- Verify Condition 3: No duplicate users in test data ---
    # "test dataにおいてユーザー重複がないこと"
    print("Verifying Condition 3...")
    assert test_df['user_id'].is_unique, "Duplicate users found in test data"
    # Also check that we have coverage (should be 2 users if both have holdout data)
    # Both users have items > 700, so both should be in test.
    assert len(test_df) == 2, f"Expected 2 test users, got {len(test_df)}"

    # --- Verify Condition 4: Test inference sequence ---
    # "test dataの推論の際に入力するシーケンスがtrain期間のデータ + test期間であれって当該期間よりも前のデータになっていること"
    print("Verifying Condition 4...")
    for _, row in test_df.iterrows():
        user_id = row['user_id']
        test_item = row['next_item']
        test_seq = parse_seq(row['seq'])
        
        # Get user's full history
        user_hist = df[df['user_id'] == user_id].sort_values('timestamp')
        full_seq = user_hist['item_id'].tolist()
        
        # Find index of test_item in full_seq
        # (Assuming unique items for simplicity in this test)
        test_idx = full_seq.index(test_item)
        
        # The input sequence should be everything before test_item
        expected_seq = full_seq[:test_idx]
        
        assert test_seq == expected_seq, f"User {user_id}: Test input sequence mismatch. Got {test_seq}, Expected {expected_seq}"

    print("All tests passed!")

if __name__ == "__main__":
    test_gts_random_split_logic()
