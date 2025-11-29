import pandas as pd
from pathlib import Path
import numpy as np

def verify_ml100k():
    data_path = Path("data/ml-100k/u1.base")
    print(f"Reading {data_path}...")
    
    # Load ratings (tab separated)
    df = pd.read_csv(data_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # Group by user
    user_sequences = df.groupby('user_id')['item_id'].apply(list)
    print(f"Total users: {len(user_sequences)}")
    
    # Test Random 90% Sampling (Hypothesis: 10% used for validation)
    print(f"\n--- Testing Random 90% Sampling on u1.base ---")
    total_samples = 0
    seq_lens = []
    
    for user_id, seq in user_sequences.items():
        if len(seq) < 3:
            continue
            
        train_indices = list(range(1, len(seq) - 1))
        
        # Randomly sample 90%
        num_to_sample = int(len(train_indices) * 0.9)
        if num_to_sample > 0:
            sampled_indices = sorted(list(np.random.choice(train_indices, num_to_sample, replace=False)))
        else:
            sampled_indices = []
            
        for i in sampled_indices:
            history = seq[:i]
            if len(history) < 3:
                continue
            seq_lens.append(min(len(history), 10))
            total_samples += 1
            
    print(f"Total Samples: {total_samples}")
    target_count = 68388
    diff = total_samples - target_count
    print(f"Diff from iLoRA ({target_count}): {diff} ({diff/target_count*100:.2f}%)")
    
    if len(seq_lens) > 0:
        print(f"Mean Seq Len: {np.mean(seq_lens):.4f}")



if __name__ == "__main__":
    verify_ml100k()
