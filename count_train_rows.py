
import pandas as pd

def count_rows():
    df = pd.read_csv("data/ml-1m/train.csv")
    # seq is space-separated string of item IDs
    # Filter: len(seq.split()) >= 3
    
    # Handle potential empty strings or NaNs
    df['seq_len'] = df['seq'].apply(lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() != '' else 0)
    
    filtered_df = df[df['seq_len'] >= 3]
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with len_seq >= 3: {len(filtered_df)}")
    print(f"Reduction: {len(df) - len(filtered_df)} rows removed")

if __name__ == "__main__":
    count_rows()
