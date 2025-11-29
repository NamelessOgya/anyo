import pandas as pd
from pathlib import Path
import numpy as np

def verify_sampling(data_dir="data/ml-1m", n_samples_list=[10, 11, 12, 13]):
    data_path = Path(data_dir) / "ratings.dat"
    print(f"Reading {data_path}...")
    
    # Load ratings
    df = pd.read_csv(data_path, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    
    # Group by user
    user_sequences = df.groupby('user_id')['item_id'].apply(list)
    print(f"Total users: {len(user_sequences)}")
    
    # Strategies to test
    strategies = [
        ("Last-N", 11),
        ("Last-N", 12),
        ("Random-N", 11),
        ("Rate", 0.076) # approx 68k / 900k
    ]

    for name, param in strategies:
        print(f"\n--- Testing Strategy: {name} ({param}) ---")
        total_samples = 0
        seq_lens = []
        
        for user_id, seq in user_sequences.items():
            if len(seq) < 3:
                continue
                
            train_indices = list(range(1, len(seq) - 1))
            
            if name == "Last-N":
                n = param
                if n > 0:
                    train_indices = train_indices[-n:]
            elif name == "Random-N":
                n = param
                if len(train_indices) > n:
                    train_indices = sorted(list(np.random.choice(train_indices, n, replace=False)))
            elif name == "Rate":
                rate = param
                # Deterministic sampling based on rate
                # e.g. take every 1/rate-th item? Or random?
                # Let's try random subset
                num_to_sample = int(len(train_indices) * rate)
                if num_to_sample > 0:
                    train_indices = sorted(list(np.random.choice(train_indices, num_to_sample, replace=False)))
                else:
                    train_indices = []

            total_samples += len(train_indices)
            
            for i in train_indices:
                history = seq[:i]
                seq_lens.append(min(len(history), 10))
                
        print(f"Total Samples: {total_samples}")
        print(f"Avg Samples/User: {total_samples / len(user_sequences):.2f}")
        
        if len(seq_lens) > 0:
            seq_lens = np.array(seq_lens)
            print(f"Seq Len (clipped 10) Mean: {seq_lens.mean():.4f}")
            print(f"Seq Len (clipped 10) Max: {seq_lens.max()}")
            print(f"Seq Len (clipped 10) Min: {seq_lens.min()}")
            
            # Percentiles
            print(f"Seq Len 25%: {np.percentile(seq_lens, 25)}")
            print(f"Seq Len 50%: {np.median(seq_lens)}")
            print(f"Seq Len 75%: {np.percentile(seq_lens, 75)}")
        
        target_count = 68388
        diff = total_samples - target_count
        print(f"Diff from iLoRA ({target_count}): {diff} ({diff/target_count*100:.2f}%)")

if __name__ == "__main__":
    verify_sampling()
