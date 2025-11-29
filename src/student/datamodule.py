import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class SASRecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len: int, num_items: int, item_id_to_name: Dict[int, str], num_candidates: int, padding_item_id: int, id_to_history_part: Dict[int, str], id_to_candidate_part: Dict[int, str], indices: Optional[List[int]] = None, subset_indices: Optional[List[int]] = None, teacher_outputs: Optional[Dict[str, torch.Tensor]] = None):
        if indices is not None:
            self.df = df.iloc[indices].reset_index(drop=True)
        else:
            self.df = df
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.padding_item_id = padding_item_id
        self.item_id_to_name = item_id_to_name
        self.num_candidates = num_candidates
        self.id_to_history_part = id_to_history_part
        self.id_to_candidate_part = id_to_candidate_part
        
        # Valid item IDs for negative sampling (exclude padding)
        self.valid_item_ids = [i for i in self.item_id_to_name.keys() if i != self.padding_item_id]

        # Partial Distillation Setup
        self.has_teacher_target = torch.zeros(len(self.df), dtype=torch.bool)
        self.teacher_targets = {} # Dict of buffers
        
        if subset_indices is not None and teacher_outputs is not None:
            # teacher_outputs: Dict[str, Tensor]
            # Check length consistency
            first_key = next(iter(teacher_outputs))
            if len(subset_indices) != len(teacher_outputs[first_key]):
                raise ValueError(f"Length mismatch: subset_indices ({len(subset_indices)}) vs teacher_outputs[{first_key}] ({len(teacher_outputs[first_key])})")
            
            # Initialize buffers for each key
            for key, tensor in teacher_outputs.items():
                # tensor shape: (num_subset, ...)
                # buffer shape: (len(df), ...)
                shape = (len(self.df),) + tensor.shape[1:]
                self.teacher_targets[key] = torch.zeros(shape, dtype=tensor.dtype)
                self.teacher_targets[key][subset_indices] = tensor
            
            self.has_teacher_target[subset_indices] = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        seq_ids = row['seq']
        next_item_id = row['next_item']
        
        seq_len = len(seq_ids)
        
        # --- Prompt Construction Parts ---
        # 1. History String
        # Use last max_seq_len items or fewer
        history_ids = seq_ids[-self.max_seq_len:] if seq_len > 0 else []
        # パディングIDを除外
        history_ids = [i for i in history_ids if i != self.padding_item_id]
        
        # Use pre-computed strings for efficiency
        history_parts = [self.id_to_history_part.get(i, str(i)) for i in history_ids]
        history_str = ", ".join(history_parts)
        
        # 2. Candidates String (Negative Sampling)
        # 履歴に含まれるアイテムと正解アイテムを除外セットに追加
        exclude_set = set(history_ids) | {next_item_id}
        
        # 除外セットに含まれないアイテムを候補プールとする
        candidates_pool = [i for i in self.valid_item_ids if i not in exclude_set]
        
        # Sample negatives
        num_negatives = self.num_candidates - 1
        # 候補プールから負例をサンプリング
        if len(candidates_pool) >= num_negatives:
            negatives = np.random.choice(candidates_pool, num_negatives, replace=False).tolist()
        else:
            # 候補が足りない場合は重複を許容してサンプリング
            negatives = np.random.choice(self.valid_item_ids, num_negatives, replace=True).tolist()
        
        # 正解アイテムと負例を合わせて候補リストを作成し、シャッフル
        candidates = negatives + [next_item_id]
        np.random.shuffle(candidates)
        
        # Use pre-computed strings for efficiency
        candidates_parts = [self.id_to_candidate_part.get(i, str(i)) for i in candidates]
        candidates_str = ", ".join(candidates_parts)
        
        # Partial Distillation Data
        has_teacher = self.has_teacher_target[index]
        teacher_target_sample = {}
        if self.teacher_targets:
            for key, buffer in self.teacher_targets.items():
                teacher_target_sample[key] = buffer[index]

        return {
            "seq_ids": seq_ids,
            "next_item_id": next_item_id,
            "history_str": history_str,
            "candidates_str": candidates_str,
            "candidates": candidates, # Return candidate IDs
            "has_teacher_target": has_teacher,
            "teacher_targets": teacher_target_sample # Dict
        }

class TeacherTrainCollater:
    """
    Teacherモデルの学習用コレーター。
    バッチ内の各サンプルに対して、iLoRA形式のプロンプトを作成し、
    トークナイズとパディングを行います。
    """
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int, padding_item_id: int, id_to_name: Dict[int, str]):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_item_id = padding_item_id
        self.id_to_name = id_to_name
        
        # iLoRAのプロンプトテンプレート
        # [HistoryEmb] はユーザーの履歴アイテム埋め込みに置換されます
        # [CansHere] は削除されました（Dense Retrievalのため）
        self.prompt_template = "This user has watched [HistoryEmb] in the previous. Please predict the next movie this user will watch."

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # --- 1. Pad sequences and gather lengths ---
        padded_seqs: List[List[int]] = []
        len_seqs: List[int] = []
        prompts: List[str] = []
        full_texts: List[str] = [] # For labels generation
        next_items: List[int] = []
        
        for sample in batch:
            seq = sample["seq_ids"]
            next_item = sample["next_item_id"]
            next_items.append(next_item) # Collect next_item_id for the final batch
            
            seq_len = len(seq)
            current_seq_len = min(seq_len, self.max_seq_len)
            len_seqs.append(current_seq_len)
            
            # Padding logic
            if seq_len < self.max_seq_len:
                padded_seq = [self.padding_item_id] * (self.max_seq_len - seq_len) + seq
            else:
                padded_seq = seq[-self.max_seq_len:]
            padded_seqs.append(padded_seq)
            
            # --- Prompt Construction ---
            # Use pre-computed strings from dataset
            # history_str = sample["history_str"] # Unused for embedding-based prompt
            # candidates_str = sample["candidates_str"] # Unused
            
            # 3. Fill Template
            # No text replacement needed for [HistoryEmb] as it is a special token
            prompt = self.prompt_template
            prompts.append(prompt)
            
            # Construct full text for Causal LM training (Prompt + Target)
            target_name = self.id_to_name.get(next_item, "")
            full_text = prompt + " " + target_name
            full_texts.append(full_text)

        # --- 2. Tokenize Prompts and Full Texts ---
        # Tokenize prompts only (to know where to mask labels)
        tokenized_prompts = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest", # Use longest to minimize padding in this intermediate step
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        
        # Tokenize full texts (Input IDs for training)
        tokenized_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512, # Adjust if needed
            add_special_tokens=True
        )
        
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        labels = input_ids.clone()
        
        # Mask the prompt part in labels
        # We assume the tokenizer produces consistent tokenization for the prefix
        # This is a simplification; for exact masking, we iterate
        for i, prompt_ids in enumerate(tokenized_prompts["input_ids"]):
            # Find the length of the prompt (excluding padding if any)
            # Note: tokenized_prompts might have padding if batch processing
            # We use the attention mask of the prompt to find valid length
            prompt_len = tokenized_prompts["attention_mask"][i].sum().item()
            
            # Mask labels up to prompt_len
            # Ensure we don't go out of bounds if full_text was truncated differently
            mask_len = min(prompt_len, labels.shape[1])
            labels[i, :mask_len] = -100
            
            # Also mask padding tokens in the full text
            labels[i][input_ids[i] == self.tokenizer.pad_token_id] = -100

        # --- 3. Collate other fields and convert to tensor ---
        # next_items = [sample["next_item_id"] for sample in batch] # Already collected in the loop
        
        candidates_batch = [sample["candidates"] for sample in batch]
        len_cans = [len(c) for c in candidates_batch]

        # --- 4. Return the final batch dictionary ---
        return {
            "seq": torch.LongTensor(padded_seqs),
            "len_seq": torch.LongTensor(len_seqs),
            "next_item": torch.LongTensor(next_items),
            "cans": torch.LongTensor(candidates_batch), # (batch_size, num_candidates)
            "len_cans": torch.LongTensor(len_cans),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels, # Added labels
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
        has_teacher_targets = []
        teacher_targets = []

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
            
            # Partial Distillation
            if "has_teacher_target" in sample:
                has_teacher_targets.append(sample["has_teacher_target"])
                teacher_targets.append(sample["teacher_targets"])

        batch_dict = {
            "seq": torch.LongTensor(padded_seqs),
            "len_seq": torch.LongTensor(len_seqs),
            "next_item": torch.LongTensor(next_items),
        }
        
        if has_teacher_targets:
            batch_dict["has_teacher_target"] = torch.tensor(has_teacher_targets, dtype=torch.bool)
            
            # Stack teacher targets
            # teacher_targets is List[Dict[str, Tensor]]
            if teacher_targets and len(teacher_targets) > 0:
                keys = teacher_targets[0].keys()
                batch_dict["teacher_targets"] = {}
                for key in keys:
                    # Stack tensors for this key
                    tensors = [t[key] for t in teacher_targets]
                    if tensors and tensors[0].numel() > 0:
                        batch_dict["teacher_targets"][key] = torch.stack(tensors)
                    else:
                        batch_dict["teacher_targets"][key] = torch.tensor([])
        
        return batch_dict


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
                 subset_indices_path: Optional[str] = None, # Added for Active Learning
                 teacher_outputs_path: Optional[str] = None, # Added for Partial Distillation
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
        self.subset_indices_path = subset_indices_path
        self.teacher_outputs_path = teacher_outputs_path
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

        # Filter sequences with length < 3 (Consistent with iLoRA reference)
        self.train_df = self.train_df[self.train_df['seq'].apply(len) >= 3]
        self.val_df = self.val_df[self.val_df['seq'].apply(len) >= 3]
        self.test_df = self.test_df[self.test_df['seq'].apply(len) >= 3]

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

        # Pre-compute prompt parts for efficiency
        # Format: "ItemName [HistoryEmb]" and "ItemName [CansEmb]"
        id_to_history_part = {
            i: f"{name} [HistoryEmb]" for i, name in self.mapped_id_to_title.items()
        }
        id_to_candidate_part = {
            i: f"{name} [CansEmb]" for i, name in self.mapped_id_to_title.items()
        }

        # Load subset indices if provided
        train_indices = None
        subset_indices = None
        teacher_outputs = None
        
        if self.subset_indices_path and os.path.exists(self.subset_indices_path):
            logger.info(f"Loading subset indices from {self.subset_indices_path}")
            loaded_indices = torch.load(self.subset_indices_path).tolist()
            
            if self.teacher_outputs_path and os.path.exists(self.teacher_outputs_path):
                # Partial Distillation Mode
                # We train on FULL dataset (train_indices = None)
                # But we pass subset_indices and teacher_outputs to dataset
                logger.info(f"Loading teacher outputs from {self.teacher_outputs_path}")
                teacher_outputs = torch.load(self.teacher_outputs_path)
                subset_indices = loaded_indices
                logger.info(f"Partial Distillation Mode: {len(subset_indices)} samples have teacher targets.")
            else:
                # Active Learning Mode (Train on Subset)
                train_indices = loaded_indices
                logger.info(f"Active Learning Mode: Training on {len(train_indices)} subset samples.")

        # The dataset now needs the item_id_to_name map and pre-computed parts
        self.train_dataset = SASRecDataset(
            self.train_df, 
            self.max_seq_len, 
            self.num_items, 
            mapped_id_to_title, 
            self.num_candidates, 
            self.padding_item_id, 
            id_to_history_part, 
            id_to_candidate_part, 
            indices=train_indices,
            subset_indices=subset_indices,
            teacher_outputs=teacher_outputs
        )
        self.val_dataset = SASRecDataset(self.val_df, self.max_seq_len, self.num_items, mapped_id_to_title, self.num_candidates, self.padding_item_id, id_to_history_part, id_to_candidate_part)
        self.test_dataset = SASRecDataset(self.test_df, self.max_seq_len, self.num_items, mapped_id_to_title, self.num_candidates, self.padding_item_id, id_to_history_part, id_to_candidate_part)

        if self.tokenizer:
            self.collater = TeacherTrainCollater(
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len,
                padding_item_id=self.padding_item_id,
                id_to_name=self.mapped_id_to_title, # Pass the mapping
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
            collate_fn=self.collater,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=self.collater,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=self.collater,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
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