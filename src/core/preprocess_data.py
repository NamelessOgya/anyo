import pandas as pd
from pathlib import Path
import argparse
import logging
import numpy as np
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_split(df: pd.DataFrame, min_seq_len: int = 3, split_method: str = 'loo', split_ratio: float = 0.8):
    logging.info(f"Sorting data by user and timestamp... (Method: {split_method})")
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    train_data, val_data, test_data = [], [], []

    if split_method == 'loo':
        logging.info("Grouping by user to create item sequences (LOO)...")
        user_sequences = df.groupby('user_id')['item_id'].apply(list)
        logging.info(f"Splitting data for {len(user_sequences)} users...")

        for user_id, seq in user_sequences.items():
            if len(seq) < min_seq_len:
                continue

            # Test data: last item in the sequence
            test_data.append({'user_id': user_id, 'seq': seq[:-1], 'next_item': seq[-1]})
            
            # Validation data: second to last item in the sequence
            val_data.append({'user_id': user_id, 'seq': seq[:-2], 'next_item': seq[-2]})

            # Training data: from the second item up to the third to last (excluding val and test targets)
            for i in range(1, len(seq) - 2):
                train_data.append({'user_id': user_id, 'seq': seq[:i], 'next_item': seq[i]})

    elif split_method == 'gts-random':
        logging.info(f"Applying Global Timestamp Split (Ratio: {split_ratio}) with Random Sampling...")
        
        # Global Split
        df_sorted = df.sort_values('timestamp')
        split_idx = int(len(df_sorted) * split_ratio)
        split_time = df_sorted.iloc[split_idx]['timestamp']
        logging.info(f"Split timestamp: {split_time}")

        # Group by user
        grouped = df_sorted.groupby('user_id')
        logging.info(f"Processing {len(grouped)} users...")

        for user_id, group in grouped:
            # Ensure sorted by timestamp (it should be, but to be safe)
            # group = group.sort_values('timestamp') 
            # (Already sorted globally, but groupby preserves order usually? 
            #  Safest to use the list we extract)
            
            seq = group['item_id'].tolist()
            timestamps = group['timestamp'].tolist()
            
            if len(seq) < min_seq_len:
                continue

            # Identify indices
            holdout_indices = [i for i, t in enumerate(timestamps) if t > split_time]
            train_indices = [i for i, t in enumerate(timestamps) if t <= split_time]

            # --- Test & Val Sampling ---
            test_idx = None
            val_idx = None

            if holdout_indices:
                # Sample Test
                test_idx = np.random.choice(holdout_indices)
                
                # Sample Val
                remaining_holdout = [i for i in holdout_indices if i != test_idx]
                if remaining_holdout:
                    val_idx = np.random.choice(remaining_holdout)
            
            # Build Test Sample
            if test_idx is not None:
                test_seq = seq[:test_idx]
                test_item = seq[test_idx]
                # Check min seq len for test? Usually we just need history.
                # If history is empty, it's cold start.
                test_data.append({'user_id': user_id, 'seq': test_seq, 'next_item': test_item})

            # Build Val Sample
            if val_idx is not None:
                val_seq = seq[:val_idx]
                val_item = seq[val_idx]
                val_data.append({'user_id': user_id, 'seq': val_seq, 'next_item': val_item})

            # --- Train Sampling ---
            # Use sliding window on TRAIN part only
            # i starts from 1 because we need at least 1 item history
            for i in train_indices:
                if i == 0: 
                    continue
                train_seq = seq[:i]
                train_item = seq[i]
                train_data.append({'user_id': user_id, 'seq': train_seq, 'next_item': train_item})

    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Convert sequence lists to string for CSV storage
    for df_to_convert in [train_df, val_df, test_df]:
        if not df_to_convert.empty:
            df_to_convert['seq'] = df_to_convert['seq'].apply(lambda x: ' '.join(map(str, x)))
            
    return train_df, val_df, test_df

def process_metadata(data_dir: Path, dataset_type: str) -> pd.DataFrame:
    """
    Reads metadata (movies/items) and standardizes it to a DataFrame with columns:
    ['item_id', 'title', 'genres']
    """
    if dataset_type == 'ml-100k':
        metadata_path = data_dir / "u.item"
        logging.info(f"Reading ML-100k metadata from {metadata_path}...")
        # ML-100k: item_id | title | release date | ... | genres (one-hot)
        # We only extract ID and Title for now, and maybe genres if needed.
        # For simplicity and consistency with ML-1M, let's just get ID and Title.
        # Genres in ML-100k are one-hot encoded at the end, which is different from ML-1M's pipe-separated string.
        # We will leave genres empty or try to reconstruct if critical. For now, ID and Title are most important.
        df = pd.read_csv(
            metadata_path,
            sep="|",
            header=None,
            usecols=[0, 1],
            names=["item_id", "title"],
            engine="python",
            encoding="latin-1",
        )
        df['genres'] = "" # Placeholder
        
    elif dataset_type == 'ml-1m':
        metadata_path = data_dir / "movies.dat"
        logging.info(f"Reading ML-1M metadata from {metadata_path}...")
        # ML-1M: item_id :: title :: genres
        df = pd.read_csv(
            metadata_path,
            sep="::",
            header=None,
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
    return df

def preprocess_data(data_dir: str, dataset_type: str, min_seq_len: int = 3, split_method: str = 'loo', split_ratio: float = 0.8):
    """
    Reads the MovieLens dataset, splits it into training, validation, and test sets
    based on user history, and saves them as CSV files. Also standardizes metadata.

    Args:
        data_dir (str): The directory containing the raw data files.
        dataset_type (str): 'ml-1m' or 'ml-100k'.
        min_seq_len (int): The minimum number of interactions a user must have to be included.
        split_method (str): 'loo' or 'gts-random'.
        split_ratio (float): Split ratio for GTS.
    """
    # Try to load config from yaml if available
    # Assuming config is at ../../../conf/dataset/{dataset_type}.yaml relative to this file?
    # Or just use the passed args. The args should be populated from yaml if the caller did so.
    # But to support "controllable via yaml" without changing caller, we can look for the yaml here.
    
    script_dir = Path(__file__).parent
    # Assuming src/core/preprocess_data.py -> conf/dataset is ../../conf/dataset
    # Adjust path resolution as needed.
    # src/core/ -> src/ -> root -> conf -> dataset
    project_root = script_dir.parent.parent
    config_path = project_root / "conf" / "dataset" / f"{dataset_type}.yaml"
    
    if config_path.exists():
        logging.info(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'split_method' in config:
                split_method = config['split_method']
                logging.info(f"Overriding split_method from config: {split_method}")
            if 'split_ratio' in config:
                split_ratio = float(config['split_ratio'])
                logging.info(f"Overriding split_ratio from config: {split_ratio}")
    
    output_dir = Path(data_dir)
    
    # 1. Process Metadata
    movies_df = process_metadata(output_dir, dataset_type)
    
    # Filter out items with "unknown" title
    logging.info("Filtering out items with 'unknown' title...")
    initial_items = len(movies_df)
    movies_df = movies_df[~movies_df['title'].str.contains("unknown", case=False, na=False)]
    removed_items = initial_items - len(movies_df)
    logging.info(f"Removed {removed_items} items.")
    
    movies_csv_path = output_dir / "movies.csv"
    logging.info(f"Saving standardized metadata to {movies_csv_path}...")
    movies_df.to_csv(movies_csv_path, index=False)
    
    valid_item_ids = set(movies_df['item_id'].unique())

    # 2. Process Interactions
    if dataset_type == 'ml-100k':
        data_path = output_dir / "u.data"
        logging.info(f"Reading ML-100k interactions from {data_path}...")
        df = pd.read_csv(data_path, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    elif dataset_type == 'ml-1m':
        data_path = output_dir / "ratings.dat"
        logging.info(f"Reading ML-1M interactions from {data_path}...")
        df = pd.read_csv(data_path, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
    # Filter interactions to keep only valid items
    logging.info("Filtering interactions to keep only valid items...")
    initial_interactions = len(df)
    df = df[df['item_id'].isin(valid_item_ids)]
    removed_interactions = initial_interactions - len(df)
    logging.info(f"Removed {removed_interactions} interactions.")

    train_df, val_df, test_df = process_and_split(df, min_seq_len, split_method, split_ratio)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    logging.info(f"Saving training data to {train_path} ({len(train_df)} rows)...")
    train_df.to_csv(train_path, index=False)

    logging.info(f"Saving validation data to {val_path} ({len(val_df)} rows)...")
    val_df.to_csv(val_path, index=False)

    logging.info(f"Saving test data to {test_path} ({len(test_df)} rows)...")
    test_df.to_csv(test_path, index=False)

    logging.info("Data preprocessing and splitting complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split MovieLens data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw data files.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=['ml-1m', 'ml-100k'], help="Type of dataset.")
    parser.add_argument("--min_seq_len", type=int, default=3, help="Minimum sequence length.")
    parser.add_argument("--split_method", type=str, default='loo', choices=['loo', 'gts-random'], help="Splitting method.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Split ratio for GTS.")
    args = parser.parse_args()
    
    preprocess_data(args.data_dir, args.dataset_type, args.min_seq_len, args.split_method, args.split_ratio)
