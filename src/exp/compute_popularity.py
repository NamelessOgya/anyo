import torch
import os
import argparse
from collections import Counter
import math

def compute_popularity(data_path, output_path, num_items):
    print(f"Computing popularity from {data_path}...")
    
    # Count item occurrences
    item_counts = Counter()
    
    import pandas as pd
    
    # Use pandas for easier CSV handling
    df = pd.read_csv(data_path)
    
    for _, row in df.iterrows():
        # seq is space separated string of item IDs
        if isinstance(row['seq'], str):
            seq_items = [int(x) for x in row['seq'].split()]
            item_counts.update(seq_items)
        
        # next_item is int
        item_counts.update([int(row['next_item'])])
            
    print(f"Total items found in data: {len(item_counts)}")
    
    # Create popularity tensor (0 is padding)
    # Indices are 1-based item IDs
    # We use log(count) as the base popularity score
    popularity = torch.zeros(num_items + 1)
    
    max_count = 0
    for item_id, count in item_counts.items():
        if item_id <= num_items:
            popularity[item_id] = count
            max_count = max(max_count, count)
            
    # Normalize? The paper says "popularity^gamma".
    # Usually popularity is count or frequency.
    # Let's store raw counts. The model can take log or power.
    # But for stability, let's store normalized frequency or just counts.
    # Paper formula: D / pop^gamma.
    # If pop is 0, division by zero.
    # So we should add smoothing?
    # Let's store raw counts, and handle smoothing/log in the model.
    # Or better, store log(count + 1) to be safe?
    # Wait, the formula is D / pop^gamma.
    # If we convert to logits (Inner Product), it's Logit + gamma * log(pop).
    # So let's store log(count + 1).
    
    # Actually, let's store raw counts to be flexible.
    # But wait, if we want to match "pop^gamma", we need pop.
    # Let's store raw counts.
    
    print(f"Max count: {max_count}")
    print(f"Saving popularity scores to {output_path}...")
    torch.save(popularity, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ml-100k/train.txt")
    parser.add_argument("--output_path", type=str, default="data/ml-100k/popularity_counts.pt")
    parser.add_argument("--num_items", type=int, default=1682)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    compute_popularity(args.data_path, args.output_path, args.num_items)
