import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Dict, Tuple
from pathlib import Path

from src.core.paths import get_project_root

class SASRecDataset(Dataset):
    def __init__(self, data: pd.DataFrame, max_seq_len: int, num_items: int, padding_item_id: int = 0):
        self.data = data
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.padding_item_id = padding_item_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        
        # シーケンスと次のアイテムIDを取得
        # iLoRAのmovielens_data.pyでは、seqはタプルのリスト、nextはタプルになっているので、IDのみ抽出
        seq_with_ratings = row['seq']
        next_item_with_rating = row['next']

        # IDのみを抽出
        item_seq_ids = [item[0] for item in seq_with_ratings]
        next_item_id = next_item_with_rating[0]

        # シーケンス長を考慮して切り詰めるかパディング
        if len(item_seq_ids) > self.max_seq_len:
            item_seq_ids = item_seq_ids[-self.max_seq_len:]
        
        seq_len = len(item_seq_ids)
        
        # パディング
        padded_item_seq = [self.padding_item_id] * self.max_seq_len
        padded_item_seq[-seq_len:] = item_seq_ids

        return {
            "item_seq": torch.tensor(padded_item_seq, dtype=torch.long),
            "item_seq_len": torch.tensor(seq_len, dtype=torch.long),
            "next_item": torch.tensor(next_item_id, dtype=torch.long)
        }

class SASRecDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_name: str = "movielens",
                 data_dir: str = "ref_repositories/iLoRA/data/ref",
                 batch_size: int = 32,
                 max_seq_len: int = 50,
                 num_workers: int = 4,
                 limit_data_rows: int = -1): # 新しい引数
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = get_project_root() / data_dir / dataset_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.padding_item_id = 0
        self.limit_data_rows = limit_data_rows # 新しい引数を保存

        self.train_data: pd.DataFrame = None
        self.val_data: pd.DataFrame = None
        self.test_data: pd.DataFrame = None
        self.item_id_to_name: Dict[int, str] = None
        self.num_items: int = 0

    def prepare_data(self):
        # データセットのダウンロードや前処理など、一度だけ実行されるべき処理
        # このプロジェクトでは、データは既に存在すると仮定
        # u.item ファイルからアイテムIDと名前のマッピングを読み込む
        item_path = self.data_dir / 'u.item'
        if not item_path.exists():
            raise FileNotFoundError(f"Item file not found: {item_path}")
        
        self.item_id_to_name = self._get_movie_id2name(item_path)
        # アイテムIDは1から始まるが、SASRecの埋め込み層は0をパディングに使うため、
        # アイテムIDを1からnum_items+1にマッピングし、0をパディングに使う
        # iLoRAのmovielens_data.pyでは、IDを0-indexedに変換している
        # (int(ll[0]) - 1)
        # そのため、ここではそのまま使用し、num_itemsは最大ID+1とする
        self.num_items = max(self.item_id_to_name.keys()) + 1


    def setup(self, stage: str = None):
        # データセットの読み込みと分割
        if stage == 'fit' or stage is None:
            self.train_data = pd.read_pickle(self.data_dir / "train_data.df")
            self.val_data = pd.read_pickle(self.data_dir / "Val_data.df")
            # iLoRAのmovielens_data.pyでは、len_seq >= 3 のデータのみを使用
            self.train_data = self.train_data[self.train_data['len_seq'] >= 3]
            self.val_data = self.val_data[self.val_data['len_seq'] >= 3]

            # データ行数制限を適用
            if self.limit_data_rows > 0:
                self.train_data = self.train_data.head(self.limit_data_rows)
                self.val_data = self.val_data.head(self.limit_data_rows)

        if stage == 'test' or stage is None:
            self.test_data = pd.read_pickle(self.data_dir / "Test_data.df")
            self.test_data = self.test_data[self.test_data['len_seq'] >= 3]
            
            # データ行数制限を適用
            if self.limit_data_rows > 0:
                self.test_data = self.test_data.head(self.limit_data_rows)

        # SASRecDatasetインスタンスの作成
        if self.train_data is not None:
            self.train_dataset = SASRecDataset(self.train_data, self.max_seq_len, self.num_items, self.padding_item_id)
        if self.val_data is not None:
            self.val_dataset = SASRecDataset(self.val_data, self.max_seq_len, self.num_items, self.padding_item_id)
        if self.test_data is not None:
            self.test_dataset = SASRecDataset(self.test_data, self.max_seq_len, self.num_items, self.padding_item_id)

    def _get_movie_id2name(self, item_path: Path) -> Dict[int, str]:
        movie_id2name = dict()
        with open(item_path, 'r', encoding="ISO-8859-1") as f:
            for l in f.readlines():
                ll = l.strip('\n').split('|')
                # iLoRAのmovielens_data.pyと同様にIDを0-indexedに変換
                movie_id2name[int(ll[0]) - 1] = self._get_mv_title(ll[1][:-7])
        return movie_id2name

    def _get_mv_title(self, s: str) -> str:
        sub_list = [", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:] + " " + s.replace(sub_s, "")
        return s

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

if __name__ == "__main__":
    # テスト
    dm = SASRecDataModule(batch_size=4, max_seq_len=50)
    dm.prepare_data()
    dm.setup()

    print(f"Number of items: {dm.num_items}")
    print(f"Item ID to Name example: {list(dm.item_id_to_name.items())[:5]}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Validation dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")

    # 最初のバッチを取得して内容を確認
    for batch in train_loader:
        print("\n--- Sample Batch from Train DataLoader ---")
        print(f"item_seq shape: {batch['item_seq'].shape}")
        print(f"item_seq_len shape: {batch['item_seq_len'].shape}")
        print(f"next_item shape: {batch['next_item'].shape}")
        print(f"item_seq (first sample): {batch['item_seq'][0]}")
        print(f"item_seq_len (first sample): {batch['item_seq_len'][0]}")
        print(f"next_item (first sample): {batch['next_item'][0]}")
        break

    print("\nSASRecDataModule test passed!")
