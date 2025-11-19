import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.core.callbacks import CustomRichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import re # Import re for regex

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging
from src.core.git_info import get_git_info

from src.student.datamodule import SASRecDataModule
from src.student.models import SASRec
from src.teacher.factory import create_teacher_model
from src.teacher.trainer_ilora import iLoRATrainer # 追加
from src.distill.trainer_distill import DistillationTrainer
from src.distill.selection_policy import AllSamplesPolicy # 現状はこれのみ
from src.student.evaluator import SASRecEvaluator # 生徒モデルの評価器
from src.distill.kd_losses import PropensityScoreCalculator # PropensityScoreCalculatorを追加
from src.distill.teacher_output_dataset import TeacherOutputDataset, teacher_output_collate_fn # Import TeacherOutputDataset and collate_fn
from torch.utils.data import DataLoader # Import DataLoader
import torch # Import torch

logger = logging.getLogger(__name__)

def find_latest_teacher_outputs_batches_dir():
    """
    Finds the latest teacher_outputs_batches directory within the result/ directory.
    """
    project_root = get_project_root()
    result_dir = project_root / "result"
    
    latest_timestamp = None
    latest_teacher_outputs_dir = None

    if not result_dir.exists():
        return None

    # Regex to match result_YYYYMMDD_HHMMSS directories
    result_dir_pattern = re.compile(r"result_(\d{8}_\d{6})")

    for entry in result_dir.iterdir():
        if entry.is_dir():
            match = result_dir_pattern.match(entry.name)
            if match:
                timestamp_str = match.group(1)
                teacher_outputs_batches_dir = entry / "teacher_outputs_batches"
                if teacher_outputs_batches_dir.is_dir():
                    if latest_timestamp is None or timestamp_str > latest_timestamp:
                        latest_timestamp = timestamp_str
                        latest_teacher_outputs_dir = teacher_outputs_batches_dir
    return latest_teacher_outputs_dir

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def run_distill(cfg: DictConfig):
    # 1. ロギング、シード、Git情報の初期化
    output_dir = get_project_root() / "result" / cfg.run.dir.split('/')[-1]
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists

    # Save the Hydra config to the experiment directory
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    setup_logging(log_dir=output_dir / "logs")
    set_seed(cfg.seed)
    git_info = get_git_info()
    logger.info(f"Git Info: {git_info}")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # 2. SASRecDataModuleの初期インスタンス化とデータ準備 (tokenizerなし)
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3. 教師モデルのインスタンス化と学習済み重みのロード
    # dmから取得したnum_itemsなどを使ってteacher_model_instanceを定義
    teacher_model_instance = None # 初期化
    if cfg.distill.teacher_checkpoint_path and cfg.distill.teacher_checkpoint_path != "???":
        logger.info(f"Loading pre-trained teacher model from {cfg.distill.teacher_checkpoint_path}")
        # iLoRATrainerのチェックポイントをロード
        loaded_teacher_trainer = iLoRATrainer.load_from_checkpoint(
            checkpoint_path=cfg.distill.teacher_checkpoint_path,
            ilora_model=create_teacher_model(
                cfg,
                num_items=dm.num_items,
                max_seq_len=cfg.student.max_seq_len,
                item_id_to_name=dm.item_id_to_name,
                padding_item_id=dm.padding_item_id,
                candidate_topk=cfg.distill.candidate_topk
            ),
            num_items=dm.num_items,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            metrics_k=cfg.eval.metrics_k,
            item_id_to_name=dm.item_id_to_name,
            strict=False
        )
        teacher_model_instance = loaded_teacher_trainer.model
        for param in teacher_model_instance.parameters():
            param.requires_grad = False
        teacher_model_instance.eval()
    else:
        logger.warning("No teacher_checkpoint_path provided or it's '???'. Using randomly initialized teacher model.")
        teacher_model_instance = create_teacher_model(
            cfg,
            num_items=dm.num_items,
            max_seq_len=cfg.student.max_seq_len,
            item_id_to_name=dm.item_id_to_name,
            padding_item_id=dm.padding_item_id,
            candidate_topk=cfg.distill.candidate_topk
        )

    # Load pre-generated teacher outputs if path is provided
    teacher_output_dataloader = None
    teacher_outputs_batches_dir_path = None

    if cfg.distill.get("teacher_outputs_batches_dir") and cfg.distill.teacher_outputs_batches_dir != "???":
        teacher_outputs_batches_dir_path = get_project_root() / cfg.distill.teacher_outputs_batches_dir
    else:
        logger.info("No teacher_outputs_batches_dir provided in config or set to '???'. Attempting to find the latest one.")
        teacher_outputs_batches_dir_path = find_latest_teacher_outputs_batches_dir()
        if teacher_outputs_batches_dir_path:
            logger.info(f"Found latest teacher outputs batches directory: {teacher_outputs_batches_dir_path}")
        else:
            logger.warning("Could not find any teacher_outputs_batches directory. Teacher outputs will be generated on-the-fly.")

    if teacher_outputs_batches_dir_path and teacher_outputs_batches_dir_path.exists():
        logger.info(f"Loading pre-generated teacher outputs from {teacher_outputs_batches_dir_path}")
        teacher_output_dataset = TeacherOutputDataset(teacher_outputs_batches_dir_path)
        teacher_output_dataloader = DataLoader(
            teacher_output_dataset,
            batch_size=1, # Each file is already a batch
            shuffle=False, # Maintain order
            num_workers=0, # Set to 0 to avoid issues with iterators across processes
            collate_fn=teacher_output_collate_fn
        )
    else:
        logger.warning("Teacher outputs batches directory not found or not specified. Teacher outputs will be generated on-the-fly.")
    # SASRecDataModuleをteacher_model_instance.tokenizerを使って再インスタンス化
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        llm_model_name=cfg.teacher.llm_model_name,
        tokenizer=teacher_model_instance.tokenizer, # ここでtokenizerを渡す
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    dm.prepare_data()
    dm.setup()

    # 3.5. 傾向スコア (Propensity Scores) の計算
    # PropensityScoreCalculatorは訓練データ全体のnext_itemを必要とする
    train_next_items = []
    for batch in dm.train_dataloader():
        train_next_items.extend(batch["next_item"].tolist())
    ps_calculator = PropensityScoreCalculator(
        item_num=dm.num_items + 1, # num_items + 1 に修正
        train_next_items=train_next_items,
        power=cfg.distill.ps_power # 新しい設定項目
    )
    propensity_scores = ps_calculator.get_ps()
    logger.info(f"Propensity scores calculated. Shape: {propensity_scores.shape}")

    # 4. 生徒モデル (SASRec) をインスタンス化
    logger.info(f"SASRec num_items: {dm.num_items}")
    logger.info(f"SASRec hidden_size: {cfg.student.hidden_size}")
    logger.info(f"SASRec max_seq_len: {cfg.student.max_seq_len}")
    logger.info(f"SASRec teacher_embedding_dim: {teacher_model_instance.llm.config.hidden_size}")
    logger.info(f"SASRec padding_item_id: {dm.padding_item_id}")
    student_model_instance = SASRec(
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
        teacher_embedding_dim=teacher_model_instance.llm.config.hidden_size,
        padding_item_id=dm.padding_item_id
    )

    # 5. DistillationTrainerのインスタンス化
    distill_trainer = DistillationTrainer(
        student_model=student_model_instance,
        teacher_model=teacher_model_instance,
        datamodule=dm,
        num_items=dm.num_items,
        ranking_loss_weight=cfg.distill.ranking_loss_weight,
        embedding_loss_weight=cfg.distill.embedding_loss_weight,
        ce_loss_weight=cfg.distill.ce_loss_weight,
        ranking_temperature=cfg.distill.ranking_temperature,
        embedding_loss_type=cfg.distill.embedding_loss_type,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k,
        selection_policy=AllSamplesPolicy(), # TODO: cfgからポリシーを生成
        gamma_position=cfg.distill.gamma_position,
        gamma_confidence=cfg.distill.gamma_confidence,
        gamma_consistency=cfg.distill.gamma_consistency,
        candidate_topk=cfg.distill.candidate_topk,
        ed_weight=cfg.distill.ed_weight,
        alpha=cfg.distill.alpha,
        beta=cfg.distill.beta,
        propensity_scores=propensity_scores,
        teacher_output_dataloader=teacher_output_dataloader # Pass the dataloader
    )

    # 6. PyTorch Lightning Trainerのインスタンス化と学習の実行
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best_distilled_student_model",
        monitor=f"val_recall@{cfg.eval.metrics_k}",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = CustomRichProgressBar()
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard_logs")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        logger=tb_logger,
        enable_checkpointing=True,
        val_check_interval=cfg.train.val_check_interval,
        log_every_n_steps=cfg.train.log_every_n_steps
    )

    logger.info("Starting distillation training...")
    trainer.fit(distill_trainer, dm.train_dataloader(), dm.val_dataloader())
    logger.info("Distillation training finished.")

    # 7. 学習済み生徒モデルの保存
    final_model_path = output_dir / "final_distilled_student_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final distilled student model saved to {final_model_path}")

    # 8. 評価の実行
    logger.info("Starting distilled student model evaluation on test set...")
    evaluation_model_path = output_dir / "final_distilled_student_model.ckpt" # 常に最終モデルをロード

    if evaluation_model_path.exists():
        logger.info(f"Loading distilled student model from {evaluation_model_path} for evaluation.")
        
        # 1. DistillationTrainerのチェックポイントをロード
        checkpoint = torch.load(evaluation_model_path, map_location='cpu')
        
        # 2. student_model_moduleのstate dictを抽出
        student_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('student_model_module.'):
                student_state_dict[key.replace('student_model_module.', '')] = value
        
        # 3. 新しいSASRecインスタンスを作成
        loaded_student_model = SASRec(
            num_items=dm.num_items,
            hidden_size=cfg.student.hidden_size,
            num_heads=cfg.student.num_heads,
            num_layers=cfg.student.num_layers,
            dropout_rate=cfg.student.dropout_rate,
            max_seq_len=cfg.student.max_seq_len,
            teacher_embedding_dim=teacher_model_instance.llm.config.hidden_size,
            padding_item_id=dm.padding_item_id
        )
        
        # 4. 抽出したstate dictをSASRecインスタンスにロード
        loaded_student_model.load_state_dict(student_state_dict)
        loaded_student_model.eval() # 評価モードに設定
        
        # 5. このSASRecインスタンスをSASRecEvaluatorに渡す
        evaluator = SASRecEvaluator(loaded_student_model, dm, metrics_k=cfg.eval.metrics_k)
    else:
        logger.warning("Final distilled student model checkpoint not found. Using current model for evaluation.")
        evaluator = SASRecEvaluator(distill_trainer.student_model_module, dm, metrics_k=cfg.eval.metrics_k)

    test_metrics = evaluator.evaluate(dm.test_dataloader())
    logger.info(f"Test Metrics: {test_metrics}")

    # 結果を保存
    with open(output_dir / "test_metrics.txt", "w") as f:
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Test metrics saved to {output_dir / 'test_metrics.txt'}")

if __name__ == "__main__":
    run_distill()


if __name__ == "__main__":
    run_distill()
