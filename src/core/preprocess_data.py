import pandas as pd
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_split(df: pd.DataFrame, min_seq_len: int = 3):
    logging.info("Sorting data by user and timestamp...")
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)

    logging.info("Grouping by user to create item sequences...")
    user_sequences = df.groupby('user_id')['item_id'].apply(list)

    train_data, val_data, test_data = [], [], []
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

def preprocess_data(data_dir: str, dataset_type: str, min_seq_len: int = 3):
    """
    Reads the MovieLens dataset, splits it into training, validation, and test sets
    based on user history, and saves them as CSV files. Also standardizes metadata.

    Args:
        data_dir (str): The directory containing the raw data files.
        dataset_type (str): 'ml-1m' or 'ml-100k'.
        min_seq_len (int): The minimum number of interactions a user must have to be included.
    """
    output_dir = Path(data_dir)
    
    # 1. Process Metadata
    movies_df = process_metadata(output_dir, dataset_type)
    movies_csv_path = output_dir / "movies.csv"
    logging.info(f"Saving standardized metadata to {movies_csv_path}...")
    movies_df.to_csv(movies_csv_path, index=False)

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

    train_df, val_df, test_df = process_and_split(df, min_seq_len)

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
    args = parser.parse_args()
    
    preprocess_data(args.data_dir, args.dataset_type, args.min_seq_len)
