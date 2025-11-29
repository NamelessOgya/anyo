import torch
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os

from src.student.datamodule import SASRecDataModule
from src.student.models import SASRec
from src.student.trainer_baseline import SASRecTrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # 1. Setup DataModule
    logger.info("Setting up DataModule...")
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size, # Use student batch size for inference
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        num_candidates=cfg.student.num_candidates,
    )
    dm.prepare_data()
    dm.setup()

    # 2. Load Student Model
    logger.info("Loading Student Model...")
    # Assuming we load from a checkpoint or initialize a new one if not provided
    # For mining, we typically need a pre-trained model.
    # Let's assume the user provides a checkpoint path in cfg.student_checkpoint_path
    
    checkpoint_path = cfg.get("student_checkpoint_path", None)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Student checkpoint not found at: {checkpoint_path}")

    # Load from checkpoint
    # We need to instantiate the model structure first or use load_from_checkpoint if arguments match
    # Here we instantiate and load weights manually or use Trainer's load_from_checkpoint
    
    # Instantiate model to get architecture
    rec_model = SASRec(
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
    )
    
    # Load Trainer/Model from checkpoint
    # Note: SASRecTrainer.load_from_checkpoint might require all init args if they are not saved in hparams
    # Safest is to load state_dict
    trainer_model = SASRecTrainer.load_from_checkpoint(
        checkpoint_path,
        rec_model=rec_model,
        num_items=dm.num_items,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=cfg.eval.metrics_k,
        strict=False # Allow missing keys if any
    )
    
    model = trainer_model.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 3. Select Strategy
    # Use active_learning config group
    strategy_name = cfg.active_learning.strategy_name
    mining_ratio = cfg.active_learning.mining_ratio
    
    logger.info(f"Using Active Learning Strategy: {strategy_name} with ratio {mining_ratio}")
    
    from src.utils.active_learning import get_strategy
    strategy = get_strategy(strategy_name, model, device, mining_ratio)
    
    # 4. Mining
    # We need a dataloader without shuffle
    mining_loader = torch.utils.data.DataLoader(
        dm.train_dataset,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False, # IMPORTANT: No shuffle to keep index order
        num_workers=cfg.train.num_workers,
        collate_fn=dm.collater,
        pin_memory=True
    )
    
    hard_indices = strategy.select_indices(mining_loader)
    
    # 5. Save Indices
    # 5. Save Indices
    output_path = Path(cfg.active_learning.hard_indices_output_path)
    torch.save(torch.tensor(hard_indices), output_path)
    logger.info(f"Saved {len(hard_indices)} indices to {output_path}")

if __name__ == "__main__":
    main()
