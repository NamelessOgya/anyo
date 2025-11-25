import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class SASRecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len: int, num_items: int, item_id_to_name: Dict[int, str]):
        self.df = df
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.padding_item_id = 0 # 0をパディング用のIDとする
        self.item_id_to_name = item_id_to_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        seq_ids = row['seq']
        next_item_id = row['next_item']
        
        # item_id_to_name is 1-indexed, so we use original IDs for lookup
        seq_names = [self.item_id_to_name.get(i, "") for i in seq_ids]

        return {
            "seq_ids": seq_ids,
            "seq_names": seq_names,
            "next_item_id": next_item_id,
        }

class TeacherTrainCollater:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int, padding_item_id: int, item_id_to_name: Dict[int, str], num_candidates: int = 20):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_item_id = padding_item_id
        self.item_id_to_name = item_id_to_name
        self.num_candidates = num_candidates
        self.prompt_template = "This user has watched [HistoryHere] in the previous. Please predict the next movie this user will watch. Choose the answer from the following 20 movie titles: [CansHere]. Answer:"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # --- 1. Pad sequences and gather lengths ---
        padded_seqs = []
        len_seqs = []
        prompts = []
        
        # Valid item IDs for negative sampling (exclude padding)
        valid_item_ids = [i for i in self.item_id_to_name.keys() if i != self.padding_item_id]

        for sample in batch:
            seq = sample["seq_ids"]
            next_item = sample["next_item_id"]
            seq_len = len(seq)
            len_seqs.append(min(seq_len, self.max_seq_len))
            
            # Padding logic
            if seq_len < self.max_seq_len:
                padded_seq = [self.padding_item_id] * (self.max_seq_len - seq_len) + seq
            else:
                padded_seq = seq[-self.max_seq_len:]
            padded_seqs.append(padded_seq)
            
            # --- Prompt Construction ---
            # 1. History String
            # Use last max_seq_len items or fewer
            history_ids = seq[-self.max_seq_len:] if seq_len > 0 else []
            history_names = [self.item_id_to_name.get(i, str(i)) for i in history_ids]
            history_str = ", ".join(history_names)
            
            # 2. Candidates String (Negative Sampling)
            # Exclude history and next item from negatives
            exclude_set = set(seq) | {next_item}
            candidates_pool = [i for i in valid_item_ids if i not in exclude_set]
            
            # Sample negatives
            num_negatives = self.num_candidates - 1
            if len(candidates_pool) >= num_negatives:
                negatives = np.random.choice(candidates_pool, num_negatives, replace=False).tolist()
            else:
                # Fallback if not enough items (should not happen in real datasets)
                negatives = np.random.choice(valid_item_ids, num_negatives, replace=True).tolist()
            
            candidates = negatives + [next_item]
            np.random.shuffle(candidates)
            
            candidates_names = [self.item_id_to_name.get(i, str(i)) for i in candidates]
            candidates_str = ", ".join(candidates_names)
            
            # 3. Fill Template
            prompt = self.prompt_template.replace("[HistoryHere]", history_str).replace("[CansHere]", candidates_str)
            prompts.append(prompt)

        # --- 2. Tokenize Prompts ---
        tokenized_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512, # Increased for text prompts
        )

        # --- 3. Collate other fields and convert to tensor ---
        next_items = [sample["next_item_id"] for sample in batch]

        # --- 4. Return the final batch dictionary ---
        return {
            "seq": torch.LongTensor(padded_seqs),
            "len_seq": torch.LongTensor(len_seqs),
            "next_item": torch.LongTensor(next_items),
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "prompts": prompts # Useful for debugging/logging
        }


class StudentCollater:
    def __init__(self, max_seq_len: int, padding_item_id: int):
        self.max_seq_len = max_seq_len
        self.padding_item_id = padding_item_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded_seqs = []
        len_seqs = []
        next_items = []

        for sample in batch:
            seq = sample["seq_ids"]
            next_items.append(sample["next_item_id"])
            seq_len = len(seq)
            len_seqs.append(min(seq_len, self.max_seq_len))
            if seq_len < self.max_seq_len:
                padded_seq = [self.padding_item_id] * (self.max_seq_len - seq_len) + seq
            else:
                padded_seq = seq[-self.max_seq_len:]
            padded_seqs.append(padded_seq)

        # The 'seq' key is used by the student model (SASRec), so we keep it.
        # We also create 'input_ids' and 'attention_mask' as dummy keys to avoid breaking
        # the teacher model which might expect them, although they are not used by the student.
        # This is a pragmatic choice to allow the same dataloader to be used in different contexts.
        return {
            "seq": torch.LongTensor(padded_seqs),
            "len_seq": torch.LongTensor(len_seqs),
            "next_item": torch.LongTensor(next_items),
        }


class SASRecDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_name: str,
                 data_dir: str,
                 batch_size: int, 
                 max_seq_len: int,
                 tokenizer: Optional[AutoTokenizer] = None, # Added tokenizer
                 num_workers: int = 0,
                 limit_data_rows: Optional[int] = None,
                 train_file: str = "train.csv",
                 val_file: str = "val.csv",
                 test_file: str = "test.csv",
                 num_candidates: int = 20, # Added num_candidates (cans_num)
                 seed: int = 42):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer # Store tokenizer
        self.num_workers = num_workers
        self.limit_data_rows = limit_data_rows
        self.padding_item_id = 0 # 0をパディング用のIDとする
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.num_candidates = num_candidates # Store num_candidates
        self.seed = seed

    def prepare_data(self):
        # データがなければダウンロードするなどの処理をここに書く
        # 今回はデータは既にある前提
        pass

    def setup(self, stage: Optional[str] = None):
        # Determine nrows for reading CSVs
        nrows_train = self.limit_data_rows if self.limit_data_rows and self.limit_data_rows > 0 else None
        # For validation and test, we usually want to evaluate on the full set, 
        # but for quick tests, we can limit them as well. Let's limit them if limit_data_rows is set.
        nrows_val_test = self.limit_data_rows if self.limit_data_rows and self.limit_data_rows > 0 else None

        # Load movie titles
        movies_df = pd.read_csv(
            self.data_dir / "movies.dat",
            sep="::",
            header=None,
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
        original_item_id_to_title = movies_df.set_index("item_id")["title"].to_dict()

        # Load pre-split dataframes with nrows
        self.train_df = pd.read_csv(self.data_dir / self.train_file, nrows=nrows_train)
        self.val_df = pd.read_csv(self.data_dir / self.val_file, nrows=nrows_val_test)
        self.test_df = pd.read_csv(self.data_dir / self.test_file, nrows=nrows_val_test)

        # Convert 'seq' column from string to list of ints
        for df in [self.train_df, self.val_df, self.test_df]:
            df['seq'] = df['seq'].apply(lambda x: [int(i) for i in x.split(' ')] if pd.notna(x) and x != '' else [])

        # Create a combined DataFrame to get all unique user and item IDs for consistent mapping
        # Now this combined_df is also limited by limit_data_rows, which is what we want for testing.
        combined_df = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        
        # Item IDとユーザーIDをリマップ
        # 0はパディングIDとして予約するため、1から開始
        # iLoRAの負例サンプリングと一致させるため、観測されたアイテムだけでなく、
        # movies.datに含まれる全てのアイテムをマッピング対象とする。
        unique_original_item_ids = sorted(movies_df['item_id'].unique().tolist())
        self.item_id_map = {original: mapped for mapped, original in enumerate(unique_original_item_ids, 1)}
        
        # Create a map from the NEW mapped IDs to original titles
        mapped_id_to_title = {
            mapped: original_item_id_to_title.get(original, str(original)) 
            for original, mapped in self.item_id_map.items()
        }
        self.mapped_id_to_title = mapped_id_to_title # Add this line

        unique_original_user_ids = sorted(list(combined_df['user_id'].unique()))
        self.user_id_map = {original: mapped for mapped, original in enumerate(unique_original_user_ids, 1)}

        # Apply mapping to all dataframes
        for df in [self.train_df, self.val_df, self.test_df]:
            df['user_id'] = df['user_id'].map(self.user_id_map)
            df['seq'] = df['seq'].apply(lambda x: [self.item_id_map.get(item) for item in x if self.item_id_map.get(item) is not None])
            df['next_item'] = df['next_item'].map(self.item_id_map)
            df.dropna(subset=['seq', 'next_item'], inplace=True) # Drop rows where mapping failed
        
        self.num_users = len(self.user_id_map)
        self.num_items = len(self.item_id_map)
        
        print("--- Data Split Statistics (limited by nrows) ---")
        print(f"Number of rows (train): {len(self.train_df)}")
        print(f"Number of rows (val): {len(self.val_df)}")
        print(f"Number of rows (test): {len(self.test_df)}")
        print("-----------------------------")

        # The dataset now needs the item_id_to_name map
        self.train_dataset = SASRecDataset(self.train_df, self.max_seq_len, self.num_items, mapped_id_to_title)
        self.val_dataset = SASRecDataset(self.val_df, self.max_seq_len, self.num_items, mapped_id_to_title)
        self.test_dataset = SASRecDataset(self.test_df, self.max_seq_len, self.num_items, mapped_id_to_title)

        if self.tokenizer:
            self.collater = TeacherTrainCollater(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                padding_item_id=self.padding_item_id,
                item_id_to_name=self.mapped_id_to_title,
                num_candidates=self.num_candidates, # Pass num_candidates
            )
        else:
            self.collater = StudentCollater(
                max_seq_len=self.max_seq_len,
                padding_item_id=self.padding_item_id,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            collate_fn=self.collater
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=self.collater
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=self.collater
        )

if __name__ == "__main__":
    # Test code
    from transformers import AutoTokenizer
    
    # Mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[HistoryEmb]']})

    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="ref_repositories/iLoRA/data/ref/movielens",
        batch_size=4,
        max_seq_len=50,
        num_workers=0,
        limit_data_rows=100,
        tokenizer=tokenizer # Pass tokenizer
    )
    dm.prepare_data()
    dm.setup()

    print("\n--- Train Dataloader Sample with Collater ---")
    for batch in dm.train_dataloader():
        print(batch.keys())
        print("seq shape:", batch['seq'].shape)
        print("input_ids shape:", batch['input_ids'].shape)
        break

    print("\n--- Val Dataloader Sample with Collater ---")
    for batch in dm.val_dataloader():
        print(batch.keys())
        break

    print(f"\nNum users: {dm.num_users}")
    print(f"Num items: {dm.num_items}")
    print(f"Padding item ID: {dm.padding_item_id}")