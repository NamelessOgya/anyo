import pytest
import torch
from src.student.models import SASRec
from src.student.datamodule import SASRecDataModule # Added import
from transformers import AutoTokenizer # Added import
from src.core.paths import get_project_root

@pytest.fixture
def sasrec_datamodule_for_models(): # Renamed to avoid name collision, and for clarity
    """
    テスト用のSASRecDataModuleを準備するフィクスチャ。
    """
    project_root = get_project_root()
    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="/workspace/data/ml-1m",
        batch_size=4,
        max_seq_len=50,
        num_workers=0
    )
    dm.prepare_data()
    dm.setup()
    return dm

@pytest.fixture
def sasrec_model_and_data(sasrec_datamodule_for_models): # Added fixture dependency
    """
    テスト用のSASRecモデルとダミーデータを準備するフィクスチャ。
    """
    num_items = sasrec_datamodule_for_models.num_items # Use num_items from datamodule
    hidden_size = 64
    num_heads = 2
    num_layers = 2
    dropout_rate = 0.1
    max_seq_len = 50
    batch_size = 4

    model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        max_seq_len=max_seq_len,
        padding_item_id=sasrec_datamodule_for_models.padding_item_id
    )

    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len))
    item_seq_len = torch.randint(1, max_seq_len + 1, (batch_size,))
    for i in range(batch_size):
        if item_seq_len[i] < max_seq_len:
            item_seq[i, item_seq_len[i]:] = sasrec_datamodule_for_models.padding_item_id # Use actual padding_item_id

    return {
        "model": model,
        "item_seq": item_seq,
        "item_seq_len": item_seq_len,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_items": num_items,
        "padding_item_id": sasrec_datamodule_for_models.padding_item_id # Pass padding_item_id
    }

def test_sasrec_forward_shape(sasrec_model_and_data):
    """
    SASRecモデルのforwardメソッドが期待される形状のテンソルを返すかテストします。
    """
    model = sasrec_model_and_data["model"]
    item_seq = sasrec_model_and_data["item_seq"]
    item_seq_len = sasrec_model_and_data["item_seq_len"]
    batch_size = sasrec_model_and_data["batch_size"]
    hidden_size = sasrec_model_and_data["hidden_size"]

    output_representation = model(item_seq, item_seq_len)
    
    assert output_representation.shape == (batch_size, hidden_size)

def test_sasrec_predict_shape(sasrec_model_and_data):
    """
    SASRecモデルのpredictメソッドが期待される形状のテンソルを返すかテストします。
    """
    model = sasrec_model_and_data["model"]
    item_seq = sasrec_model_and_data["item_seq"]
    item_seq_len = sasrec_model_and_data["item_seq_len"]
    batch_size = sasrec_model_and_data["batch_size"]
    num_items = sasrec_model_and_data["num_items"]

    prediction_scores = model.predict(item_seq, item_seq_len)
    
    assert prediction_scores.shape == (batch_size, num_items)

def test_sasrec_forward_with_teacher_embedding(sasrec_model_and_data, sasrec_datamodule_for_models):
    """
    SASRecモデルのforwardメソッドが教師の埋め込みを受け取った場合に正しく動作するかテストします。
    """
    num_items = sasrec_model_and_data["num_items"]
    hidden_size = sasrec_model_and_data["hidden_size"]
    batch_size = sasrec_model_and_data["batch_size"]
    item_seq = sasrec_model_and_data["item_seq"]
    item_seq_len = sasrec_model_and_data["item_seq_len"]
    padding_item_id = sasrec_datamodule_for_models.padding_item_id
    
    teacher_embedding_dim = 128
    ed_weight = 0.5

    model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50,
        teacher_embedding_dim=teacher_embedding_dim,
        ed_weight=ed_weight,
        padding_item_id=padding_item_id
    )

    teacher_embeddings = torch.randn(batch_size, teacher_embedding_dim)

    output_representation = model(item_seq, item_seq_len, teacher_embeddings=teacher_embeddings)
    
    assert output_representation.shape == (batch_size, teacher_embedding_dim)
