import pytest
import torch
import torch.nn as nn # Added import
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.teacher.ilora_model import iLoRAModel
from src.teacher.trainer_ilora import iLoRATrainer
from src.student.datamodule import SASRecDataModule # データモジュールは共通
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.teacher.mlp_projector import MLPProjector # Added import # Added import

@pytest.fixture(scope="module")
def ilora_trainer_and_data():
    """
    テスト用のiLoRATrainerとダミーデータを準備するフィクスチャ。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # iLoRAModelのインスタンス化 (ダミー設定)
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 24,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })
    # LLMとTokenizerをロード
    llm = AutoModelForCausalLM.from_pretrained(ilora_cfg.llm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ilora_cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # データモジュール
    dm = SASRecDataModule(
        dataset_name="movielens",
        data_dir="data/ml-1m",
        batch_size=2, 
        max_seq_len=20, 
        num_workers=0,
        tokenizer=tokenizer # Pass tokenizer
    )
    dm.prepare_data()
    dm.setup()

    # ダミーのrec_modelとprojectorを作成
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size_rec, num_items_rec, padding_item_id): # Add padding_item_id
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items_rec + 2, hidden_size_rec, padding_idx=padding_item_id) # Adjust size and padding_idx
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size_rec).to(x.device)
    dummy_rec_model = DummyRecModel(ilora_cfg.hidden_size, dm.num_items, dm.padding_item_id).to(device) # Pass padding_item_id
    dummy_projector = MLPProjector(
        input_dim=ilora_cfg.hidden_size,
        output_dim=llm.config.hidden_size,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate
    ).to(device)

    ilora_model_instance = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=ilora_cfg.num_lora_experts,
        lora_r=ilora_cfg.lora_r,
        lora_alpha=ilora_cfg.lora_alpha,
        lora_dropout=ilora_cfg.lora_dropout,
        num_items=dm.num_items,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate,
        rec_model=dummy_rec_model,
        projector=dummy_projector,
        candidate_topk=10
    ).to(device)
    # iLoRATrainerのインスタンス化
    trainer_model = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=1e-4,
        weight_decay=0.01,
        metrics_k=10,
        item_id_to_name=dm.item_id_to_name # 追加
    ).to(device)

    return {
        "trainer_model": trainer_model,
        "datamodule": dm
    }

def test_ilora_training_step(ilora_trainer_and_data):
    """
    iLoRATrainerのtraining_stepが正しく動作するかテストします。
    """
    trainer_model = ilora_trainer_and_data["trainer_model"]
    dm = ilora_trainer_and_data["datamodule"]

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu", # テスト用にCPUを使用
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1 # 1バッチのみ実行
    )

    # training_stepを実行
    trainer.fit(trainer_model, dm.train_dataloader())
    
    # 損失がスカラーであり、nanやinfでないことを確認
    assert isinstance(trainer_model.trainer.callback_metrics["train_loss_step"], torch.Tensor)
    assert not torch.isnan(trainer_model.trainer.callback_metrics["train_loss_step"])
    assert not torch.isinf(trainer_model.trainer.callback_metrics["train_loss_step"])

    # 教師モデルのパラメータが更新されていないことを確認 (勾配がNoneまたはゼロ)
    # iLoRAModelのパラメータはpeftによってラップされているため、直接アクセスが難しい場合がある
    # ここでは、ilora_model_instanceのパラメータが更新されていないことを確認する
    # (DistillationTrainerの__init__でteacher_model.eval()しているため、勾配計算されないはず)
    # Check a specific base LLM parameter that should be frozen
    base_llm_param = trainer_model.model.llm.model.decoder.layers[0].self_attn.q_proj.weight
    assert not base_llm_param.requires_grad or (base_llm_param.grad is None or torch.all(base_llm_param.grad == 0))