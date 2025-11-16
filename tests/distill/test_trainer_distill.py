import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.student.models import SASRec
from src.teacher.ilora_model import iLoRAModel
from src.distill.trainer_distill import DistillationTrainer
from src.student.datamodule import SASRecDataModule
from src.distill.selection_policy import AllSamplesPolicy

@pytest.fixture(scope="module")
def distill_trainer_and_data():
    """
    テスト用のDistillationTrainerとダミーデータを準備するフィクスチャ。
    """
    # データモジュール
    dm = SASRecDataModule(batch_size=2, max_seq_len=20, num_workers=0)
    dm.prepare_data()
    dm.setup()

    # 生徒モデルのインスタンス化
    student_model_instance = SASRec(
        num_users=1000, # ダミー
        num_items=dm.num_items,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=20
    )

    # 教師モデルのインスタンス化 (ダミー設定)
    teacher_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })
    teacher_model_instance = iLoRAModel(
        llm_model_name=teacher_cfg.llm_model_name,
        num_lora_experts=teacher_cfg.num_lora_experts,
        lora_r=teacher_cfg.lora_r,
        lora_alpha=teacher_cfg.lora_alpha,
        lora_dropout=teacher_cfg.lora_dropout,
        num_items=dm.num_items,
        max_seq_len=dm.max_seq_len,
        hidden_size=teacher_cfg.hidden_size,
        dropout_rate=teacher_cfg.dropout_rate,
        device="cpu" # 明示的にCPUを指定
    )

    # 蒸留トレーナーのインスタンス化
    distill_trainer = DistillationTrainer(
        student_model=student_model_instance,
        teacher_model=teacher_model_instance,
        num_items=dm.num_items,
        ranking_loss_weight=1.0,
        embedding_loss_weight=1.0,
        ce_loss_weight=1.0,
        ranking_temperature=2.0,
        embedding_loss_type="mse",
        learning_rate=1e-3,
        weight_decay=0.01,
        metrics_k=10,
        selection_policy=AllSamplesPolicy()
    )

    return {
        "trainer_model": distill_trainer,
        "datamodule": dm
    }

def test_distill_training_step(distill_trainer_and_data):
    """
    DistillationTrainerのtraining_stepが正しく動作するかテストします。
    """
    trainer_model = distill_trainer_and_data["trainer_model"]
    dm = distill_trainer_and_data["datamodule"]

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
    assert isinstance(trainer_model.trainer.callback_metrics["train_total_loss_step"], torch.Tensor)
    assert not torch.isnan(trainer_model.trainer.callback_metrics["train_total_loss_step"])
    assert not torch.isinf(trainer_model.trainer.callback_metrics["train_total_loss_step"])

    # 教師モデルのパラメータが更新されていないことを確認 (勾配がNoneまたはゼロ)
    # iLoRAModelのパラメータはpeftによってラップされているため、直接アクセスが難しい場合がある
    # ここでは、ilora_model_instanceのパラメータが更新されていないことを確認する
    # (DistillationTrainerの__init__でteacher_model.eval()しているため、勾配計算されないはず)
    for param in trainer_model.teacher_model.parameters():
        if param.requires_grad:
            assert param.grad is None or torch.all(param.grad == 0)
