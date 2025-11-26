import pandas as pd
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(data_dir: str, min_seq_len: int = 3):
    """
    Reads the MovieLens 1M dataset, splits it into training, validation, and test sets
    based on user history, and saves them as CSV files.

    Args:
        data_dir (str): The directory containing the 'ratings.dat' file.
        min_seq_len (int): The minimum number of interactions a user must have to be included.
    """
    data_path = Path(data_dir) / "ratings.dat"
    output_dir = Path(data_dir)
    
    logging.info(f"Reading data from {data_path}...")
    df = pd.read_csv(data_path, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')

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

        # Training data: from the second item up to the second to last
        for i in range(1, len(seq) - 1):
            train_data.append({'user_id': user_id, 'seq': seq[:i], 'next_item': seq[i]})

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    # Convert sequence lists to string for CSV storage
    for df_to_convert in [train_df, val_df, test_df]:
        df_to_convert['seq'] = df_to_convert['seq'].apply(lambda x: ' '.join(map(str, x)))

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
    parser = argparse.ArgumentParser(description="Preprocess and split MovieLens 1M data.")
    parser.add_argument("--data_dir", type=str, default="data/ml-1m", help="Directory containing the ratings.dat file.")
    args = parser.parse_args()
    
    preprocess_data(args.data_dir)
