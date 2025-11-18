import pytest
import torch
import torch.nn as nn # For DummyRecModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector

@pytest.fixture(scope="module")
def ilora_model_and_data():
    """
    テスト用のiLoRAModelとダミーデータを準備するフィクスチャ。
    scope="module" にすることで、このモジュールのテスト全体で一度だけ実行される。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_model_name = "facebook/opt-125m"
    num_lora_experts = 3
    lora_r = 24
    lora_alpha = 16
    lora_dropout = 0.05
    num_items = 1000
    max_seq_len = 20
    hidden_size = 64
    dropout_rate = 0.1
    batch_size = 2

    # LLMとTokenizerをロード
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # ダミーのrec_modelとprojectorを作成
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size_rec, num_items_rec):
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items_rec + 1, hidden_size_rec)
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size_rec).to(x.device)
    
    dummy_rec_model = DummyRecModel(hidden_size, num_items).to(device)
    dummy_projector = MLPProjector(
        input_dim=hidden_size,
        output_dim=llm.config.hidden_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    ).to(device)

    model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=num_lora_experts,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_items=num_items,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        rec_model=dummy_rec_model,
        projector=dummy_projector,
        candidate_topk=10  # Add dummy value
    ).to(device)

    # ダミーのバッチデータを作成
    input_ids = torch.randint(0, len(tokenizer), (batch_size, max_seq_len)).to(device)
    attention_mask = torch.ones((batch_size, max_seq_len), dtype=torch.long).to(device)
    seq = torch.randint(1, num_items, (batch_size, max_seq_len)).to(device)
    len_seq = torch.randint(1, max_seq_len + 1, (batch_size,)).to(device)
    cans = torch.randint(1, num_items, (batch_size, 20)).to(device) # 20は仮の候補数
    len_cans = torch.randint(1, 21, (batch_size,)).to(device)
    item_id = torch.randint(1, num_items, (batch_size,)).to(device)
    next_item = torch.randint(1, num_items, (batch_size,)).to(device)

    # tokensをBatchEncodingオブジェクトに変換
    tokens_batch_encoding = BatchEncoding({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    batch = {
        "tokens": tokens_batch_encoding,
        "seq": seq,
        "len_seq": len_seq,
        "cans": cans,
        "len_cans": len_cans,
        "item_id": item_id,
        "next_item": next_item
    }

    return {
        "model": model,
        "batch": batch,
        "batch_size": batch_size,
        "num_items": num_items,
        "hidden_size": llm.config.hidden_size,
        "num_lora_experts": num_lora_experts
    }

def test_ilora_forward_shape(ilora_model_and_data):
    """
    iLoRAModelのforwardメソッドが期待される形状のテンソルを返すかテストします。
    """
    model = ilora_model_and_data["model"]
    batch = ilora_model_and_data["batch"]
    batch_size = ilora_model_and_data["batch_size"]
    # num_items = ilora_model_and_data["num_items"] # Not directly used for raw LLM output shape
    max_seq_len = 20 # Defined in fixture
    
    output_scores = model(batch)
    
    # iLoRAModel.forward returns CausalLMOutputWithPast object
    # Check logits shape
    assert output_scores.logits.shape == (batch_size, max_seq_len, len(model.tokenizer))
    # Check last hidden state shape
    assert output_scores.hidden_states[-1].shape == (batch_size, max_seq_len, model.llm.config.hidden_size)

def test_ilora_get_teacher_outputs_shape(ilora_model_and_data):
    """
    get_teacher_outputsメソッドが返す辞書の各要素が期待される形状であるかテストします。
    """
    model = ilora_model_and_data["model"]
    batch = ilora_model_and_data["batch"]
    batch_size = ilora_model_and_data["batch_size"]
    num_items = ilora_model_and_data["num_items"]
    hidden_size = ilora_model_and_data["hidden_size"]

    teacher_outputs = model.get_teacher_outputs(batch)
    
    assert "ranking_scores" in teacher_outputs
    assert "embeddings" in teacher_outputs
    assert teacher_outputs["ranking_scores"].shape == (batch_size, num_items)
    assert teacher_outputs["embeddings"].shape == (batch_size, hidden_size)
