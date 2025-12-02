import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import logging
from pathlib import Path
import pandas as pd

from src.student.datamodule import SASRecDataModule
from src.teacher.factory import create_teacher_model
from src.teacher.bigrec_model import BigRecModel
from src.teacher.moe_bigrec_model import MoEBigRecModel
from src.core.seed import set_seed
from src.core.logger import setup_logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed)
    
    # Setup Logging
    # output_dir = Path(cfg.run.dir) / "tuning"
    # output_dir.mkdir(parents=True, exist_ok=True)
    # setup_logging(log_dir=output_dir)
    
    # 1. Load DataModule
    print("Loading DataModule...")
    from transformers import AutoTokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name, use_fast=False)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token_id = 0
        llm_tokenizer.pad_token = llm_tokenizer.decode(0)
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_tokenizer.padding_side = "left"

    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.teacher.get("batch_size", 32),
        max_seq_len=cfg.student.max_seq_len,
        tokenizer=llm_tokenizer,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv",
        seed=cfg.seed
    )
    dm.prepare_data()
    dm.setup()
    
    # 2. Load Model
    # We need to find the checkpoint.
    # Assuming the user runs this with the same experiment config, we can look in the output dir.
    # Or the user provides a checkpoint path via `ckpt_path=...`
    
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        # Try to find in default output dir
        output_dir = Path(cfg.run.dir)
        checkpoints = list((output_dir / "checkpoints").glob("*.ckpt"))
        if checkpoints:
            # Sort by modification time or name? Usually we want 'last' or 'best'.
            # Let's pick 'last.ckpt' if exists, else the most recent one.
            last_ckpt = output_dir / "checkpoints" / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
            else:
                # Sort by modification time descending
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                ckpt_path = str(checkpoints[0])
            print(f"Auto-detected checkpoint: {ckpt_path}")
        else:
            raise ValueError("No checkpoint found. Please provide `ckpt_path=/path/to/ckpt`")
    else:
        print(f"Using provided checkpoint: {ckpt_path}")

    # Determine model class
    model_type = cfg.teacher.get("model_type", "bigrec")
    if model_type == "bigrec":
        ModelClass = BigRecModel
    elif model_type == "moe_bigrec":
        ModelClass = MoEBigRecModel
    else:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")

    print(f"Loading {model_type} from {ckpt_path}...")
    # We load with strict=False because sometimes config params in checkpoint might differ slightly or we want to override
    # But load_from_checkpoint uses hparams from checkpoint if available.
    # We want to use the current config for things like item_embeddings_path if it changed?
    # Actually, load_from_checkpoint merges args.
    
    # Important: We need to pass the arguments that are NOT in hparams or need to be refreshed (like item_id_to_name mapping which is not saved in hparams usually, or is it?)
    # hparams usually saves simple types. item_id_to_name is a dict, might be saved.
    # But let's pass it to be safe.
    
    model = ModelClass.load_from_checkpoint(
        ckpt_path,
        map_location="cpu", # Load to CPU first
        strict=False,
        # Override params if needed
        item_id_to_name=dm.mapped_id_to_title,
        item_embeddings_path=cfg.teacher.item_embeddings_path,
        popularity_path=cfg.teacher.popularity_path
    )
    
    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 3. Tuning Loop
    lambdas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    # Allow override via config
    if cfg.get("tuning_lambdas"):
        lambdas = cfg.tuning_lambdas
        
    results = []
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False, # Disable logger to avoid creating many log files
        enable_checkpointing=False
    )
    
    print("\nStarting Tuning...")
    print(f"Lambdas to test: {lambdas}")
    
    for lam in lambdas:
        print(f"\nEvaluating with popularity_lambda = {lam}...")
        model.popularity_lambda = lam
        
        # Run Validation
        metrics = trainer.validate(model, datamodule=dm, verbose=False)
        # metrics is a list of dicts (one per dataloader)
        metric_dict = metrics[0]
        
        # Extract key metrics
        hr = metric_dict.get(f"val_hr@{cfg.eval.metrics_k}", 0.0)
        ndcg = metric_dict.get(f"val_ndcg@{cfg.eval.metrics_k}", 0.0)
        
        print(f"Result: HR@{cfg.eval.metrics_k}={hr:.4f}, NDCG@{cfg.eval.metrics_k}={ndcg:.4f}")
        
        results.append({
            "lambda": lam,
            "hr": hr,
            "ndcg": ndcg
        })

    # 4. Report
    df = pd.DataFrame(results)
    print("\n--- Tuning Results ---")
    print(df.sort_values("ndcg", ascending=False))
    
    best_res = df.loc[df["ndcg"].idxmax()]
    print(f"\nBest Lambda: {best_res['lambda']} (NDCG: {best_res['ndcg']:.4f})")
    
    # Save results
    output_path = Path("tuning_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path.absolute()}")

if __name__ == "__main__":
    main()
