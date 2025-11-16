import pytest
import torch
from src.student.models import SASRec

@pytest.fixture
def sasrec_model_and_data():
    """
    テスト用のSASRecモデルとダミーデータを準備するフィクスチャ。
    """
    num_users = 100
    num_items = 5000
    hidden_size = 64
    num_heads = 2
    num_layers = 2
    dropout_rate = 0.1
    max_seq_len = 50
    batch_size = 4

    model = SASRec(
        num_users=num_users,
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        max_seq_len=max_seq_len
    )

    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len))
    item_seq_len = torch.randint(1, max_seq_len + 1, (batch_size,))
    for i in range(batch_size):
        if item_seq_len[i] < max_seq_len:
            item_seq[i, item_seq_len[i]:] = 0

    return {
        "model": model,
        "item_seq": item_seq,
        "item_seq_len": item_seq_len,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_items": num_items
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
    
    assert prediction_scores.shape == (batch_size, num_items + 1)
