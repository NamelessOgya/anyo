import pytest
import torch
from transformers import AutoTokenizer # Added import
from src.student.datamodule import SASRecDataModule

@pytest.fixture
def sasrec_datamodule():
    """
    テスト用のSASRecDataModuleを準備するフィクスチャ。
    """
    # ダミーのtokenizerを作成
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})

    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="data/ml-1m",
        batch_size=4,
        max_seq_len=50,
        num_workers=0, # テスト時は0に設定
        limit_data_rows=1000, # Limit rows for testing
        tokenizer=tokenizer # Pass tokenizer
    )
    dm.prepare_data()
    dm.setup()
    return dm

def test_datamodule_setup(sasrec_datamodule):
    """
    SASRecDataModuleのsetupメソッドが正しく動作するかテストします。
    """
    dm = sasrec_datamodule
    
    # データセットがロードされているか確認
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    
    # num_itemsが正しく計算されているか確認 (movielensは1682アイテム)
    assert dm.num_items > 0

def test_dataloader_batch_shape(sasrec_datamodule):
    """
    DataLoaderが返すバッチの形状が正しいかテストします。
    """
    dm = sasrec_datamodule
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    batch_size = dm.batch_size
    max_seq_len = dm.max_seq_len
    
    assert "seq" in batch
    assert "len_seq" in batch
    assert "next_item" in batch
    
    assert batch["seq"].shape == (batch_size, max_seq_len)
    assert batch["len_seq"].shape == (batch_size,)
    assert batch["next_item"].shape == (batch_size,)
    
    assert batch["seq"].dtype == torch.long
    assert batch["len_seq"].dtype == torch.long
    assert batch["next_item"].dtype == torch.long

def test_dataloader_padding(sasrec_datamodule):
    """
    バッチ内のitem_seqが正しくパディングされているかテストします。
    """
    dm = sasrec_datamodule
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    item_seq = batch["seq"]
    item_seq_len = batch["len_seq"]
    
    for i in range(dm.batch_size):
        seq_len = item_seq_len[i].item()
        padding_part = item_seq[i, :-seq_len]
        data_part = item_seq[i, -seq_len:]
        
        # パディング部分がすべて0であることを確認
        assert torch.all(padding_part == dm.padding_item_id)
        # データ部分に0（パディングID）が含まれていないことを確認
        assert not torch.any(data_part == dm.padding_item_id)
