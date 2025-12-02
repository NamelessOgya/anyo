import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_dummy_ml100k_data(output_dir: str, num_users: int = 20, num_items: int = 100, num_interactions: int = 500):
    """
    Creates a dummy ML-100k dataset structure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating dummy ML-100k data in {output_dir}...")

    # 1. u.data (user id | item id | rating | timestamp)
    user_ids = np.random.randint(1, num_users + 1, num_interactions)
    item_ids = np.random.randint(1, num_items + 1, num_interactions)
    ratings = np.random.randint(1, 6, num_interactions)
    timestamps = np.random.randint(874724710, 893286638, num_interactions)

    df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Save as u.data (tab separated)
    df.to_csv(output_dir / "u.data", sep='\t', index=False, header=False)
    
    # 2. u.item (movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western)
    # We only need movie id and title mostly
    genres = [0] * 19
    items = []
    for i in range(1, num_items + 1):
        items.append([i, f"Movie {i} (1995)", "01-Jan-1995", "", "http://imdb.com"] + genres)
        
    df_item = pd.DataFrame(items)
    # Save as u.item (pipe separated, latin-1 encoding usually)
    df_item.to_csv(output_dir / "u.item", sep='|', index=False, header=False, encoding='latin-1')

    print("Dummy data created successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    create_dummy_ml100k_data(args.output_dir)
