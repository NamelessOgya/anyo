import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.teacher.ilora_model import iLoRAModel
from src.teacher.trainer_ilora import iLoRATrainer
from src.student.datamodule import SASRecDataModule # データモジュールは共通

@pytest.fixture(scope="module")
def ilora_trainer_and_data():
    """
    テスト用のiLoRATrainerとダミーデータを準備するフィクスチャ。
    """
    # データモジュール
    dm = SASRecDataModule(batch_size=2, max_seq_len=20, num_workers=0)
    dm.prepare_data()
    dm.setup()

    # iLoRAModelのインスタンス化 (ダミー設定)
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })
    ilora_model_instance = iLoRAModel(
        llm_model_name=ilora_cfg.llm_model_name,
        num_lora_experts=ilora_cfg.num_lora_experts,
        lora_r=ilora_cfg.lora_r,
        lora_alpha=ilora_cfg.lora_alpha,
        lora_dropout=ilora_cfg.lora_dropout,
        num_items=dm.num_items,
        max_seq_len=dm.max_seq_len,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate,
        device="cpu" # 明示的にCPUを指定
    )

    # iLoRATrainerのインスタンス化
    trainer_model = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=1e-4,
        weight_decay=0.01,
        metrics_k=10
    )

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
