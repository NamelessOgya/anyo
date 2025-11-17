import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.student.models import SASRec
from src.teacher.ilora_model import iLoRAModel
from src.distill.trainer_distill import DistillationTrainer
from src.student.datamodule import SASRecDataModule
from src.distill.selection_policy import AllSamplesPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.teacher.mlp_projector import MLPProjector
from src.distill.kd_losses import PropensityScoreCalculator # Import PropensityScoreCalculator

@pytest.fixture(scope="module")
def distill_trainer_and_data():
    """
    テスト用のDistillationTrainerとダミーデータを準備するフィクスチャ。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 教師モデルのインスタンス化 (ダミー設定)
    teacher_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 24,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })
    # LLMとTokenizerをロード
    llm = AutoModelForCausalLM.from_pretrained(teacher_cfg.llm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(teacher_cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # データモジュール
    dm = SASRecDataModule(
        batch_size=2, 
        max_seq_len=20, 
        num_workers=0,
        tokenizer=tokenizer # Pass tokenizer
    )
    dm.prepare_data()
    dm.setup()

    # Propensity Scoresの計算
    train_next_items = []
    for batch in dm.train_dataloader():
        train_next_items.extend(batch["next_item"].tolist())
    ps_calculator = PropensityScoreCalculator(
        item_num=dm.num_items + 1, # num_items + 1 に修正
        train_next_items=train_next_items,
        power=0.05 # ダミー値
    )
    propensity_scores = ps_calculator.get_ps()

    # 生徒モデルのインスタンス化 (訓練対象)
    student_model_instance = SASRec(
        num_items=dm.num_items,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=20,
        teacher_embedding_dim=llm.config.hidden_size
    ).to(device)
    student_model_instance.train()

    # 教師モデル用のダミーの推薦モデル (パラメータはフリーズされる)
    dummy_rec_model = SASRec(
        num_items=dm.num_items,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=20,
        teacher_embedding_dim=llm.config.hidden_size
    ).to(device)
    dummy_rec_model.eval() # 評価モード
    for param in dummy_rec_model.parameters():
        param.requires_grad = False

    teacher_model_instance = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=teacher_cfg.num_lora_experts,
        lora_r=teacher_cfg.lora_r,
        lora_alpha=teacher_cfg.lora_alpha,
        lora_dropout=teacher_cfg.lora_dropout,
        num_items=dm.num_items,
        hidden_size=teacher_cfg.hidden_size,
        dropout_rate=teacher_cfg.dropout_rate,
        rec_model=dummy_rec_model, # ダミーのrec_modelを渡す
        projector=MLPProjector(
            input_dim=student_model_instance.hidden_size,
            output_dim=llm.config.hidden_size,
            hidden_size=teacher_cfg.hidden_size,
            dropout_rate=teacher_cfg.dropout_rate
        ),
        candidate_topk=10
    )
    # 蒸留トレーナーのインスタンス化
    distill_trainer_no_dro = DistillationTrainer(
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
        selection_policy=AllSamplesPolicy(),
        gamma_position=1.0,
        gamma_confidence=1.0,
        gamma_consistency=1.0,
        candidate_topk=10,
        ed_weight=0.1,
        alpha=0.0, # DRO無効
        beta=1.0,
        propensity_scores=propensity_scores
    )

    distill_trainer_with_dro = DistillationTrainer(
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
        selection_policy=AllSamplesPolicy(),
        gamma_position=1.0,
        gamma_confidence=1.0,
        gamma_consistency=1.0,
        candidate_topk=10,
        ed_weight=0.1,
        alpha=0.5, # DRO有効
        beta=1.0,
        propensity_scores=propensity_scores
    )

    return {
        "trainer_model_no_dro": distill_trainer_no_dro,
        "trainer_model_with_dro": distill_trainer_with_dro,
        "datamodule": dm
    }

def test_distill_training_step_no_dro(distill_trainer_and_data):
    """
    DistillationTrainerのtraining_stepがDROなしで正しく動作するかテストします。
    """
    trainer_model = distill_trainer_and_data["trainer_model_no_dro"]
    dm = distill_trainer_and_data["datamodule"]

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1
    )

    trainer.fit(trainer_model, dm.train_dataloader())
    
    assert isinstance(trainer_model.trainer.callback_metrics["train_total_loss"], torch.Tensor)
    assert not torch.isnan(trainer_model.trainer.callback_metrics["train_total_loss"])
    assert not torch.isinf(trainer_model.trainer.callback_metrics["train_total_loss"])
    
    # DRO関連のログがないことを確認
    assert "train_ce_dro_loss" not in trainer_model.trainer.callback_metrics

    for param in trainer_model.teacher_model.parameters():
        if param.requires_grad:
            assert param.grad is None or torch.all(param.grad == 0)

def test_distill_training_step_with_dro(distill_trainer_and_data):
    """
    DistillationTrainerのtraining_stepがDROありで正しく動作するかテストします。
    """
    trainer_model = distill_trainer_and_data["trainer_model_with_dro"]
    dm = distill_trainer_and_data["datamodule"]

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1
    )

    trainer.fit(trainer_model, dm.train_dataloader())
    
    assert isinstance(trainer_model.trainer.callback_metrics["train_total_loss"], torch.Tensor)
    assert not torch.isnan(trainer_model.trainer.callback_metrics["train_total_loss"])
    assert not torch.isinf(trainer_model.trainer.callback_metrics["train_total_loss"])

    # DRO関連のログがあることを確認
    assert "train_ce_dro_loss" in trainer_model.trainer.callback_metrics
    assert not torch.isnan(trainer_model.trainer.callback_metrics["train_ce_dro_loss"])
    assert not torch.isinf(trainer_model.trainer.callback_metrics["train_ce_dro_loss"])

    for param in trainer_model.teacher_model.parameters():
        if param.requires_grad:
            assert param.grad is None or torch.all(param.grad == 0)

def test_distill_training_step_loss_difference(distill_trainer_and_data):
    """
    DROの有無で損失が異なることを確認します。
    """
    trainer_model_no_dro = distill_trainer_and_data["trainer_model_no_dro"]
    trainer_model_with_dro = distill_trainer_and_data["trainer_model_with_dro"]
    dm = distill_trainer_and_data["datamodule"]

    # DROなしで学習
    trainer_no_dro = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1
    )
    trainer_no_dro.fit(trainer_model_no_dro, dm.train_dataloader())
    loss_no_dro = trainer_model_no_dro.trainer.callback_metrics["train_total_loss"]

    # DROありで学習
    trainer_with_dro = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1
    )
    trainer_with_dro.fit(trainer_model_with_dro, dm.train_dataloader())
    loss_with_dro = trainer_model_with_dro.trainer.callback_metrics["train_total_loss"]

    # 損失が異なることを確認 (厳密な比較は難しいので、単に等しくないことを確認)
    assert not torch.isclose(loss_no_dro, loss_with_dro, atol=1e-6)
