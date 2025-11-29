import pandas as pd
import pickle

try:
    df = pd.read_pickle("temp_train_data.df")
    print("Columns:", df.columns)
    print("Total rows:", len(df))
    
    if 'user_id' in df.columns:
        num_users = df['user_id'].nunique()
        print("Num users:", num_users)
        print("Avg samples per user:", len(df) / num_users)
        
        # Check samples for a few users
        print("\nSample user data:")
        print(df[df['user_id'] == df['user_id'].iloc[0]].sort_values('len_seq'))
    else:
        print("No 'user_id' column. Showing first 5 rows:")
        print(df.head())
        
    # Check max item ID to identify dataset (ML-100k vs ML-1M)
    max_item_id = 0
    # Sample a subset for speed if needed, but full scan is safer
    for seq in df['seq']:
        # seq is list of tuples (id, rating)
        for item in seq:
            if item[0] > max_item_id:
                max_item_id = item[0]
                
    print(f"Max Item ID: {max_item_id}")
    if max_item_id < 2000:
        print("Likely MovieLens-100k (Max ID ~1682)")
    else:
        print("Likely MovieLens-1M (Max ID ~3952)")


except Exception as e:
    print(f"Error reading pickle: {e}")
