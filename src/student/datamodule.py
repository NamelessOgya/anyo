import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Any, Dict, List, Tuple
from pathlib import Path

from src.core.paths import get_project_root

class SASRecDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame, 
                 max_seq_len: int, 
                 num_items: int, 
                 padding_item_id: int = 0,
                 tokenizer: Any = None, # Added tokenizer
                 max_gen_length: int = 64, # Added max_gen_length
                 item_id_to_name: Dict[int, str] = None): # Added item_id_to_name
        self.data = data
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.padding_item_id = padding_item_id
        self.tokenizer = tokenizer
        self.max_gen_length = max_gen_length
        self.item_id_to_name = item_id_to_name

        # Special token IDs
        self.his_token_id = self.tokenizer.additional_special_tokens_ids[self.tokenizer.additional_special_tokens.index('[HistoryEmb]')] if self.tokenizer and '[HistoryEmb]' in self.tokenizer.additional_special_tokens else None
        self.cans_token_id = self.tokenizer.additional_special_tokens_ids[self.tokenizer.additional_special_tokens.index('[CansEmb]')] if self.tokenizer and '[CansEmb]' in self.tokenizer.additional_special_tokens else None
        self.item_token_id = self.tokenizer.additional_special_tokens_ids[self.tokenizer.additional_special_tokens.index('[ItemEmb]')] if self.tokenizer and '[ItemEmb]' in self.tokenizer.additional_special_tokens else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        seq_with_ratings = row['seq']
        next_item_with_rating = row['next']

        item_seq_ids = [item[0] for item in seq_with_ratings]
        next_item_id = next_item_with_rating[0]

        if len(item_seq_ids) > self.max_seq_len:
            item_seq_ids = item_seq_ids[-self.max_seq_len:]
        
        seq_len = len(item_seq_ids)
        
        padded_item_seq = [self.padding_item_id] * self.max_seq_len
        padded_item_seq[-seq_len:] = item_seq_ids

        tokens = None
        if self.tokenizer and self.item_id_to_name: # Only construct prompt if tokenizer and item_id_to_name are available
            # プロンプトの構築
            # [HistoryEmb] item_name_1, item_name_2, ... [ItemEmb] next_item_name
            history_names = [self.item_id_to_name.get(item_id, "[UNK]") for item_id in item_seq_ids]
            next_item_name = self.item_id_to_name.get(next_item_id, "[UNK]")

            # MInterfaceのプロンプト構造を参考に、特殊トークンを埋め込む
            # ここでは簡略化のため、[HistoryEmb]と[ItemEmb]のみを使用
            # [CansEmb]は別途生成する必要があるが、ここではダミーで対応
            prompt_text = f"{self.tokenizer.decode(self.his_token_id)} {', '.join(history_names)} {self.tokenizer.decode(self.item_token_id)} {next_item_name}"
            
            # プロンプトをトークナイズ
            tokens = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_gen_length # max_gen_lengthを使用
            )
            
            # tokensはバッチの次元がないのでsqueeze
            # input_ids = tokens["input_ids"].squeeze(0)
            # attention_mask = tokens["attention_mask"].squeeze(0)
        
        # cans (候補アイテム) の生成
        # iLoRAのmovielens_data.pyでは、ネガティブサンプリングでcansを生成している
        # ここでは簡略化のため、ランダムなアイテムを生成
        num_cans = 20 # 仮の候補数
        cans_ids = torch.randint(1, self.num_items, (num_cans,)).tolist()
        # next_item_idがcansに含まれていない場合は追加
        if next_item_id not in cans_ids:
            cans_ids[0] = next_item_id # 最初の候補を正解アイテムにする
        
        # cansのパディング
        padded_cans = [self.padding_item_id] * num_cans
        padded_cans[:len(cans_ids)] = cans_ids
        len_cans = len(cans_ids)

        return_dict = {
            "seq": torch.tensor(padded_item_seq, dtype=torch.long),
            "len_seq": torch.tensor(seq_len, dtype=torch.long),
            "cans": torch.tensor(padded_cans, dtype=torch.long),
            "len_cans": torch.tensor(len_cans, dtype=torch.long),
            "item_id": torch.tensor(next_item_id, dtype=torch.long), # next_item_idをitem_idとして渡す
            "next_item": torch.tensor(next_item_id, dtype=torch.long)
        }
        if tokens is not None:
            return_dict["tokens"] = tokens
        
        return return_dict

class SASRecDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_name: str = "movielens",
                 data_dir: str = "ref_repositories/iLoRA/data/ref/movielens", # Updated default
                 batch_size: int = 32,
                 max_seq_len: int = 50,
                 num_workers: int = 4,
                 limit_data_rows: int = -1,
                 llm_model_name: str = None, # Added llm_model_name
                 tokenizer: Any = None, # Added tokenizer
                 max_gen_length: int = 64): # Added max_gen_length
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = get_project_root() / data_dir # Removed dataset_name append
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.padding_item_id = 0
        self.limit_data_rows = limit_data_rows

        self.llm_model_name = llm_model_name
        self.tokenizer = tokenizer
        self.max_gen_length = max_gen_length

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
        self.num_items = max(self.item_id_to_name.keys())
        self.padding_item_id = self.num_items + 1 # Set padding_item_id here

        # トークナイザーが渡されていない場合、LLMモデル名からロード
        if self.tokenizer is None and self.llm_model_name:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # 特殊トークンを追加
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})



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
            self.train_dataset = SASRecDataset(
                self.train_data, 
                self.max_seq_len, 
                self.num_items, 
                self.padding_item_id,
                tokenizer=self.tokenizer, # Pass tokenizer
                max_gen_length=self.max_gen_length, # Pass max_gen_length
                item_id_to_name=self.item_id_to_name # Pass item_id_to_name
            )
        if self.val_data is not None:
            self.val_dataset = SASRecDataset(
                self.val_data, 
                self.max_seq_len, 
                self.num_items, 
                self.padding_item_id,
                tokenizer=self.tokenizer, # Pass tokenizer
                max_gen_length=self.max_gen_length, # Pass max_gen_length
                item_id_to_name=self.item_id_to_name # Pass item_id_to_name
            )
        if self.test_data is not None:
            self.test_dataset = SASRecDataset(
                self.test_data, 
                self.max_seq_len, 
                self.num_items, 
                self.padding_item_id,
                tokenizer=self.tokenizer, # Pass tokenizer
                max_gen_length=self.max_gen_length, # Pass max_gen_length
                item_id_to_name=self.item_id_to_name # Pass item_id_to_name
            )

    def _get_movie_id2name(self, item_path: Path) -> Dict[int, str]:
        movie_id2name = dict()
        with open(item_path, 'r', encoding="ISO-8859-1") as f:
            for l in f.readlines():
                ll = l.strip('\n').split('|')
                # iLoRAのmovielens_data.pyと同様にIDを0-indexedに変換
                movie_id2name[int(ll[0])] = self._get_mv_title(ll[1][:-7])
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

    def calculate_propensity_score(self) -> torch.Tensor:
        """
        訓練データから各アイテムの傾向スコアを計算します。
        """
        if self.train_data is None:
            raise RuntimeError("Train data is not loaded. Call setup() first.")

        pop_dict = {}
        for seq_with_ratings in self.train_data['seq']:
            for item_id, rating in seq_with_ratings:
                if item_id in pop_dict:
                    pop_dict[item_id] += 1
                else:
                    pop_dict[item_id] = 1
        
        # アイテムIDの最大値まで考慮
        pop = torch.zeros(self.num_items + 1)
        for item_id, count in pop_dict.items():
            pop[item_id] = count

        ps = pop + 1
        ps = ps / torch.sum(ps)
        ps = torch.pow(ps, 0.05)
        return ps

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # テスト用のLLMとTokenizerをロード
    llm_model_name_test = "facebook/opt-125m"
    tokenizer_test = AutoTokenizer.from_pretrained(llm_model_name_test)
    if tokenizer_test.pad_token is None:
        tokenizer_test.pad_token = tokenizer_test.eos_token
    tokenizer_test.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    
    # テスト
    dm = SASRecDataModule(
        batch_size=4, 
        max_seq_len=50, 
        llm_model_name=llm_model_name_test,
        tokenizer=tokenizer_test,
        max_gen_length=64
    )
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
        print(f"tokens input_ids shape: {batch['tokens']['input_ids'].shape}")
        print(f"tokens attention_mask shape: {batch['tokens']['attention_mask'].shape}")
        print(f"seq shape: {batch['seq'].shape}")
        print(f"len_seq shape: {batch['len_seq'].shape}")
        print(f"cans shape: {batch['cans'].shape}")
        print(f"len_cans shape: {batch['len_cans'].shape}")
        print(f"item_id shape: {batch['item_id'].shape}")
        print(f"next_item shape: {batch['next_item'].shape}")
        
        print(f"tokens input_ids (first sample): {batch['tokens']['input_ids'][0]}")
        print(f"tokens attention_mask (first sample): {batch['tokens']['attention_mask'][0]}")
        print(f"seq (first sample): {batch['seq'][0]}")
        print(f"len_seq (first sample): {batch['len_seq'][0]}")
        print(f"cans (first sample): {batch['cans'][0]}")
        print(f"len_cans (first sample): {batch['len_cans'][0]}")
        print(f"item_id (first sample): {batch['item_id'][0]}")
        print(f"next_item (first sample): {batch['next_item'][0]}")
        break

    print("\nSASRecDataModule test passed!")
