import pytest
import torch
from src.student.datamodule import SASRecDataModule

@pytest.fixture
def sasrec_datamodule():
    """
    テスト用のSASRecDataModuleを準備するフィクスチャ。
    """
    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="ref_repositories/iLoRA/data/ref",
        batch_size=4,
        max_seq_len=50,
        num_workers=0 # テスト時は0に設定
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
    assert dm.num_items == 1682

def test_dataloader_batch_shape(sasrec_datamodule):
    """
    DataLoaderが返すバッチの形状が正しいかテストします。
    """
    dm = sasrec_datamodule
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    batch_size = dm.batch_size
    max_seq_len = dm.max_seq_len
    
    assert "item_seq" in batch
    assert "item_seq_len" in batch
    assert "next_item" in batch
    
    assert batch["item_seq"].shape == (batch_size, max_seq_len)
    assert batch["item_seq_len"].shape == (batch_size,)
    assert batch["next_item"].shape == (batch_size,)
    
    assert batch["item_seq"].dtype == torch.long
    assert batch["item_seq_len"].dtype == torch.long
    assert batch["next_item"].dtype == torch.long

def test_dataloader_padding(sasrec_datamodule):
    """
    バッチ内のitem_seqが正しくパディングされているかテストします。
    """
    dm = sasrec_datamodule
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    item_seq = batch["item_seq"]
    item_seq_len = batch["item_seq_len"]
    
    for i in range(dm.batch_size):
        seq_len = item_seq_len[i].item()
        padding_part = item_seq[i, :-seq_len]
        data_part = item_seq[i, -seq_len:]
        
        # パディング部分がすべて0であることを確認
        assert torch.all(padding_part == dm.padding_item_id)
        # データ部分に0（パディングID）が含まれていないことを確認
        assert not torch.any(data_part == dm.padding_item_id)
