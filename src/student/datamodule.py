import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Dict
import numpy as np
from pathlib import Path

class SASRecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len: int, num_items: int):
        self.df = df
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.padding_item_id = 0 # 0をパディング用のIDとする

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        seq = row['seq']
        next_item = row['next_item']
        
        seq_len = len(seq)
        
        # シーケンスの長さをmax_seq_lenに合わせる
        if seq_len < self.max_seq_len:
            # パディング
            padded_seq = [self.padding_item_id] * (self.max_seq_len - seq_len) + seq
        else:
            # 切り捨て
            padded_seq = seq[-self.max_seq_len:]
        
        # アイテムIDは1-indexedなので、0-indexedに変換
        # ただし、パディングIDは0のまま
        padded_seq = [i - 1 if i > 0 else 0 for i in padded_seq]
        next_item_0_indexed = next_item - 1

        return {
            "seq": torch.LongTensor(padded_seq),
            "len_seq": torch.LongTensor([min(seq_len, self.max_seq_len)]),
            "next_item": torch.LongTensor([next_item_0_indexed])
        }

class SASRecDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_name: str,
                 data_dir: str,
                 batch_size: int, 
                 max_seq_len: int,
                 num_workers: int = 0,
                 limit_data_rows: Optional[int] = None,
                 train_file: str = "train.csv", # New parameter
                 val_file: str = "val.csv",     # New parameter
                 test_file: str = "test.csv"):  # New parameter
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.limit_data_rows = limit_data_rows
        self.padding_item_id = 0 # 0をパディング用のIDとする
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

    def prepare_data(self):
        # データがなければダウンロードするなどの処理をここに書く
        # 今回はデータは既にある前提
        pass

    def setup(self, stage: Optional[str] = None):
        # Load pre-split dataframes
        self.train_df = pd.read_csv(self.data_dir / self.train_file)
        self.val_df = pd.read_csv(self.data_dir / self.val_file)
        self.test_df = pd.read_csv(self.data_dir / self.test_file)

        # Convert 'seq' column from string to list of ints
        for df in [self.train_df, self.val_df, self.test_df]:
            df['seq'] = df['seq'].apply(lambda x: [int(i) for i in x.split(' ')] if pd.notna(x) and x != '' else [])

        # Create a combined DataFrame to get all unique user and item IDs for consistent mapping
        combined_df = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        
        # Extract all item IDs from 'seq' and 'next_item'
        all_item_ids = []
        for seq_list in combined_df['seq']:
            all_item_ids.extend(seq_list)
        all_item_ids.extend(combined_df['next_item'].tolist())
        
        # Item IDとユーザーIDをリマップ
        # 0はパディングIDとして予約するため、1から開始
        unique_original_item_ids = sorted(list(set(all_item_ids)))
        self.item_id_map = {original: mapped for mapped, original in enumerate(unique_original_item_ids, 1)}
        self.item_id_to_name = {mapped: original for original, mapped in self.item_id_map.items()}
        
        unique_original_user_ids = sorted(list(combined_df['user_id'].unique()))
        self.user_id_map = {original: mapped for mapped, original in enumerate(unique_original_user_ids, 1)}

        # Apply mapping to all dataframes
        for df in [self.train_df, self.val_df, self.test_df]:
            df['user_id'] = df['user_id'].map(self.user_id_map)
            df['seq'] = df['seq'].apply(lambda x: [self.item_id_map[item] for item in x])
            df['next_item'] = df['next_item'].map(self.item_id_map)
        
        self.num_users = len(self.user_id_map)
        self.num_items = len(self.item_id_map)

        if self.limit_data_rows:
            self.train_df = self.train_df.head(self.limit_data_rows)
            # valとtestはデータ数を制限しない（比較のため）

        print("--- Data Split Statistics ---")
        print(f"Number of rows (train): {len(self.train_df)}")
        print(f"Number of rows (val): {len(self.val_df)}")
        print(f"Number of rows (test): {len(self.test_df)}")
        print("-----------------------------")

        self.train_dataset = SASRecDataset(self.train_df, self.max_seq_len, self.num_items)
        self.val_dataset = SASRecDataset(self.val_df, self.max_seq_len, self.num_items)
        self.test_dataset = SASRecDataset(self.test_df, self.max_seq_len, self.num_items)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

if __name__ == "__main__":
    # テストコード
    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="ref_repositories/iLoRA/data/ref/movielens",
        batch_size=4,
        max_seq_len=50,
        num_workers=0,
        limit_data_rows=1000
    )
    dm.prepare_data()
    dm.setup()

    print("\n--- Train Dataloader Sample ---")
    for batch in dm.train_dataloader():
        print(batch)
        break

    print("\n--- Val Dataloader Sample ---")
    for batch in dm.val_dataloader():
        print(batch)
        break

    print("\n--- Test Dataloader Sample ---")
    for batch in dm.test_dataloader():
        print(batch)
        break

    print(f"\nNum users: {dm.num_users}")
    print(f"Num items: {dm.num_items}")
    print(f"Padding item ID: {dm.padding_item_id}")