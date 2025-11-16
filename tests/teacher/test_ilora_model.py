import pytest
import torch
from src.teacher.ilora_model import iLoRAModel

@pytest.fixture(scope="module")
def ilora_model_and_data():
    """
    テスト用のiLoRAModelとダミーデータを準備するフィクスチャ。
    scope="module" にすることで、このモジュールのテスト全体で一度だけ実行される。
    """
    llm_model_name = "facebook/opt-125m"
    num_lora_experts = 3
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    num_items = 1000
    max_seq_len = 20
    hidden_size = 64
    dropout_rate = 0.1
    batch_size = 2
    padding_item_id = 0

    dummy_item_id_to_name = {i: f"Item {i}" for i in range(num_items + 1)}
    model = iLoRAModel(
        llm_model_name=llm_model_name,
        num_lora_experts=num_lora_experts,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        item_id_to_name=dummy_item_id_to_name,
        padding_item_id=padding_item_id
    )

    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len)).to(model.device)
    item_seq_len = torch.randint(1, max_seq_len + 1, (batch_size,)).to(model.device)

    return {
        "model": model,
        "item_seq": item_seq,
        "item_seq_len": item_seq_len,
        "batch_size": batch_size,
        "num_items": num_items,
        "hidden_size": hidden_size,
        "num_lora_experts": num_lora_experts
    }

def test_ilora_forward_shape(ilora_model_and_data):
    """
    iLoRAModelのforwardメソッドが期待される形状のテンソルを返すかテストします。
    """
    model = ilora_model_and_data["model"]
    item_seq = ilora_model_and_data["item_seq"]
    item_seq_len = ilora_model_and_data["item_seq_len"]
    batch_size = ilora_model_and_data["batch_size"]
    num_items = ilora_model_and_data["num_items"]

    output_scores = model(item_seq, item_seq_len)
    
    assert output_scores.shape == (batch_size, num_items + 1)

def test_ilora_get_teacher_outputs_shape(ilora_model_and_data):
    """
    get_teacher_outputsメソッドが返す辞書の各要素が期待される形状であるかテストします。
    """
    model = ilora_model_and_data["model"]
    item_seq = ilora_model_and_data["item_seq"]
    item_seq_len = ilora_model_and_data["item_seq_len"]
    batch_size = ilora_model_and_data["batch_size"]
    num_items = ilora_model_and_data["num_items"]
    hidden_size = ilora_model_and_data["hidden_size"]

    teacher_outputs = model.get_teacher_outputs(item_seq, item_seq_len)
    
    assert "ranking_scores" in teacher_outputs
    assert "embeddings" in teacher_outputs
    assert teacher_outputs["ranking_scores"].shape == (batch_size, num_items + 1)
    assert teacher_outputs["embeddings"].shape == (batch_size, hidden_size)

def test_gating_network(ilora_model_and_data):
    """
    ゲーティングネットワークの出力が期待通りであるかテストします。
    """
    model = ilora_model_and_data["model"]
    item_seq = ilora_model_and_data["item_seq"]
    item_seq_len = ilora_model_and_data["item_seq_len"]
    batch_size = ilora_model_and_data["batch_size"]
    num_lora_experts = ilora_model_and_data["num_lora_experts"]

    h_seq = model._get_sequence_representation(item_seq, item_seq_len)
    expert_weights = model.gating_network(h_seq)
    
    # 形状の確認
    assert expert_weights.shape == (batch_size, num_lora_experts)
    # Softmaxの出力なので、合計が1になることを確認
    assert torch.allclose(torch.sum(expert_weights, dim=1), torch.tensor(1.0))
