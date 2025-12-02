import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.teacher.bigrec_model import BigRecModel
from src.student.models import SASRec
from src.student.trainer_baseline import SASRecTrainer
from src.student.datamodule import SASRecDataModule, StudentCollater
from src.data.collators import BigRecCollator
from src.core.seed import set_seed
from ensemble_model import EnsembleBigRecSASRec, AlphaNetwork

logger = logging.getLogger(__name__)

class EnsembleDataset(Dataset):
    def __init__(self, original_dataset, bigrec_embs):
        self.original_dataset = original_dataset
        self.bigrec_embs = bigrec_embs
        
    def __len__(self):
        return len(self.original_dataset)
        
    def __getitem__(self, idx):
        return {
            "original": self.original_dataset[idx],
            "bigrec_emb": self.bigrec_embs[idx]
        }

class EnsembleCollator:
    def __init__(self, student_collator):
        self.student_collator = student_collator
        
    def __call__(self, batch):
        originals = [b["original"] for b in batch]
        bigrec_embs = [b["bigrec_emb"] for b in batch]
        
        batch_out = self.student_collator(originals)
        batch_out["bigrec_emb"] = torch.stack(bigrec_embs)
        
        return batch_out

def precompute_bigrec_embeddings(model, dataloader, device, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path)
        
    logger.info("Pre-computing BigRec embeddings...")
    embeddings = []
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Generate
            # Logic adapted from BigRecModel._evaluate_step
            generated_ids = model.model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=model.hparams.max_target_length,
                num_beams=model.num_beams,
                num_return_sequences=1,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                early_stopping=True,
                do_sample=False # Deterministic for pre-computation
            )
            
            input_len = batch["prompt_input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            
            generated_texts = model.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            generated_texts = [text.strip('"') for text in generated_texts]
            
            # Embed
            text_inputs = model.tokenizer(
                generated_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.hparams.max_target_length
            ).to(device)
            
            with model.model.disable_adapter():
                text_outputs = model.model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    output_hidden_states=True
                )
                last_hidden = text_outputs.hidden_states[-1]
                pred_embeddings = last_hidden[:, -1, :] # (B, Dim)
                
            embeddings.append(pred_embeddings.cpu())
            
    all_embs = torch.cat(embeddings)
    
    if cache_path:
        torch.save(all_embs, cache_path)
        logger.info(f"Saved embeddings to {cache_path}")
        
    return all_embs

@hydra.main(version_base=None, config_path="../../conf", config_name="post_training_config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    # 1. Load DataModule
    logger.info("Loading DataModule...")
    # We need tokenizer for BigRecCollator
    from transformers import AutoTokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name, use_fast=False)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token_id = 0
        llm_tokenizer.pad_token = llm_tokenizer.decode(0)
    llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm_tokenizer.padding_side = "left"

    # 1. Load DataModule
    logger.info("Loading DataModule...")
    data_dir = hydra.utils.to_absolute_path(cfg.dataset.data_dir)
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=data_dir,
        batch_size=cfg.teacher.batch_size, # Use teacher batch size for generation/inference
        max_seq_len=cfg.student.max_seq_len,
        limit_data_rows=cfg.dataset.limit_data_rows,
    )
    dm.prepare_data()
    dm.setup()
    
    # 2. Load BigRec Model
    logger.info("Loading BigRec Model...")
    ckpt_path = cfg.post_training.ckpt_path
    if not ckpt_path:
        # Try to find default
        output_dir = Path(cfg.run.dir) / "checkpoints"
        if output_dir.exists():
            ckpts = list(output_dir.glob("*.ckpt"))
            if ckpts:
                ckpts.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                ckpt_path = str(ckpts[0])
    
    if not ckpt_path:
        raise ValueError("No BigRec checkpoint found. Please provide post_training.ckpt_path=...")
        
    bigrec_model = BigRecModel.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=False,
        item_id_to_name=dm.mapped_id_to_title,
        item_embeddings_path=cfg.teacher.item_embeddings_path,
        popularity_path=cfg.teacher.popularity_path
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. Create BigRec Collator
    bigrec_collator = BigRecCollator(
        tokenizer=bigrec_model.tokenizer,
        item_id_to_name=dm.mapped_id_to_title,
        max_source_length=cfg.teacher.max_source_length,
        max_target_length=cfg.teacher.max_target_length,
        use_cot=cfg.teacher.get("use_cot", False),
        max_history_items=cfg.teacher.get("max_history_items", 20),
        sasrec_max_seq_len=cfg.student.max_seq_len
    )
    
    # 4. Phase 1: Tune Gamma
    logger.info("--- Phase 1: Tuning Population Gamma ---")
    
    # Create Val Loader (No Shuffle)
    val_loader = DataLoader(
        dm.val_dataset,
        batch_size=cfg.teacher.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=bigrec_collator
    )
    
    # Pre-compute Val Embeddings
    val_embs_path = Path("val_bigrec_embs.pt")
    val_bigrec_embs = precompute_bigrec_embeddings(bigrec_model, val_loader, device, cache_path=val_embs_path)
    
    # Tuning Loop
    lambdas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    best_ndcg = -1.0
    best_lambda = 0.0
    
    # We need item embeddings and popularity scores on device
    item_embs = bigrec_model.item_embeddings.to(device)
    pop_scores = bigrec_model.popularity_scores.to(device) if bigrec_model.popularity_scores is not None else None
    
    # Get targets from val dataset
    val_targets = torch.tensor(dm.val_df['next_item'].values).to(device)
    
    # Move val_bigrec_embs to device in chunks if needed, but for tuning let's try full batch if fits
    # Or iterate
    
    for lam in lambdas:
        logger.info(f"Testing lambda={lam}")
        
        # Compute metrics
        # We can do this in batches to save memory
        hits = 0
        ndcg = 0
        total = 0
        
        batch_size = 100 # Internal batch size for metric calculation
        num_samples = len(val_bigrec_embs)
        
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            batch_embs = val_bigrec_embs[i:end].to(device)
            batch_targets = val_targets[i:end]
            
            dists = torch.cdist(batch_embs.float(), item_embs.float(), p=2)
            max_dist = dists.max(dim=1, keepdim=True)[0]
            dists = dists / (max_dist + 1e-8)
            
            if pop_scores is not None and lam > 0:
                pop_factor = (pop_scores + 1.0) ** lam
                dists = dists / pop_factor.unsqueeze(0)
                
            # Top-K
            _, topk_indices = torch.topk(dists, k=10, dim=1, largest=False)
            
            # Metrics
            for j in range(len(batch_targets)):
                t = batch_targets[j].item()
                preds = topk_indices[j].tolist()
                if t in preds:
                    hits += 1
                    rank = preds.index(t)
                    ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
            
            total += (end - i)
            
        hr_score = hits / total
        ndcg_score = ndcg / total
        logger.info(f"Lambda={lam}: HR@10={hr_score:.4f}, NDCG@10={ndcg_score:.4f}")
        
        if ndcg_score > best_ndcg:
            best_ndcg = ndcg_score
            best_lambda = lam
            
    logger.info(f"Best Lambda: {best_lambda} (NDCG: {best_ndcg:.4f})")
    
    # 5. Phase 2: Pre-compute Training Embeddings
    logger.info("--- Phase 2: Pre-computing Training Embeddings ---")
    train_loader_gen = DataLoader(
        dm.train_dataset,
        batch_size=cfg.teacher.batch_size,
        shuffle=False, # Must be False to align with dataset
        num_workers=cfg.train.num_workers,
        collate_fn=bigrec_collator
    )
    
    train_embs_path = Path("train_bigrec_embs.pt")
    train_bigrec_embs = precompute_bigrec_embeddings(bigrec_model, train_loader_gen, device, cache_path=train_embs_path)
    
    # 6. Phase 3: Train Ensemble
    logger.info("--- Phase 3: Training Ensemble ---")
    
    # Load SASRec
    sasrec_ckpt = cfg.post_training.sasrec_ckpt
    if not sasrec_ckpt:
        # Try default student path
        # Assuming structure: experiments/student/sasrec/checkpoints/last.ckpt
        # But we are in a run dir?
        # Let's assume user provides it or we guess.
        # User said: experiments/student/sasrec/checkpoints/last.ckpt
        sasrec_ckpt = "experiments/student/sasrec/checkpoints/last.ckpt"
        if not os.path.exists(sasrec_ckpt):
             # Try relative to project root
             sasrec_ckpt = str(Path(__file__).resolve().parents[2] / "experiments/student/sasrec/checkpoints/last.ckpt")
    
    if not os.path.exists(sasrec_ckpt):
         raise ValueError(f"SASRec checkpoint not found at {sasrec_ckpt}")
         
    logger.info(f"Loading SASRec from {sasrec_ckpt}")
    
    # Instantiate SASRec
    sasrec_model = SASRec(
        num_items=dm.num_items,
        hidden_size=cfg.student.hidden_size,
        num_heads=cfg.student.num_heads,
        num_layers=cfg.student.num_layers,
        dropout_rate=cfg.student.dropout_rate,
        max_seq_len=cfg.student.max_seq_len,
        padding_item_id=dm.padding_item_id
    )
    
    # Load weights via Trainer wrapper
    sasrec_trainer = SASRecTrainer.load_from_checkpoint(
        sasrec_ckpt,
        rec_model=sasrec_model,
        num_items=dm.num_items,
        learning_rate=cfg.student.learning_rate,
        weight_decay=cfg.student.weight_decay,
        metrics_k=cfg.eval.metrics_k
    )
    
    # Init Ensemble
    alpha_net = AlphaNetwork(input_dim=cfg.student.hidden_size)
    ensemble_model = EnsembleBigRecSASRec(
        sasrec_model=sasrec_trainer.model,
        alpha_net=alpha_net,
        item_embeddings=bigrec_model.item_embeddings,
        popularity_scores=bigrec_model.popularity_scores,
        popularity_lambda=best_lambda,
        lr=1e-3
    )
    
    # Create Ensemble Dataloader
    ensemble_dataset = EnsembleDataset(dm.train_dataset, train_bigrec_embs)
    student_collator = StudentCollater(max_seq_len=cfg.student.max_seq_len, padding_item_id=dm.padding_item_id)
    ensemble_collator = EnsembleCollator(student_collator)
    
    ensemble_loader = DataLoader(
        ensemble_dataset,
        batch_size=cfg.student.batch_size, # Use student batch size for training alpha
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=ensemble_collator
    )
    
    # Val Loader for Ensemble
    # We need val_bigrec_embs aligned with val_dataset
    ensemble_val_dataset = EnsembleDataset(dm.val_dataset, val_bigrec_embs)
    ensemble_val_loader = DataLoader(
        ensemble_val_dataset,
        batch_size=cfg.student.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=ensemble_collator
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=5, # Short training for alpha
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=True,
        enable_checkpointing=True,
        default_root_dir="ensemble_result"
    )
    
    trainer.fit(ensemble_model, train_dataloaders=ensemble_loader, val_dataloaders=ensemble_val_loader)
    
    # 7. Phase 4: Test
    logger.info("--- Phase 4: Testing ---")
    
    # Pre-compute Test Embeddings
    test_loader_gen = DataLoader(
        dm.test_dataset,
        batch_size=cfg.teacher.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=bigrec_collator
    )
    
    test_embs_path = Path("test_bigrec_embs.pt")
    test_bigrec_embs = precompute_bigrec_embeddings(bigrec_model, test_loader_gen, device, cache_path=test_embs_path)
    
    # --- Evaluate BigRec (Optimized Gamma) ---
    logger.info(f"Evaluating BigRec (Lambda={best_lambda})...")
    test_targets = torch.tensor(dm.test_df['next_item'].values).to(device)
    
    hits = 0
    ndcg = 0
    total = 0
    batch_size = 100
    num_samples = len(test_bigrec_embs)
    
    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_embs = test_bigrec_embs[i:end].to(device)
        batch_targets = test_targets[i:end]
        
        dists = torch.cdist(batch_embs.float(), item_embs.float(), p=2)
        max_dist = dists.max(dim=1, keepdim=True)[0]
        dists = dists / (max_dist + 1e-8)
        
        if pop_scores is not None and best_lambda > 0:
            pop_factor = (pop_scores + 1.0) ** best_lambda
            dists = dists / pop_factor.unsqueeze(0)
            
        _, topk_indices = torch.topk(dists, k=10, dim=1, largest=False)
        
        for j in range(len(batch_targets)):
            t = batch_targets[j].item()
            preds = topk_indices[j].tolist()
            if t in preds:
                hits += 1
                rank = preds.index(t)
                ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
        
        total += (end - i)
        
    bigrec_hr = hits/total
    bigrec_ndcg = ndcg/total
    logger.info(f"BigRec Test Result: HR@10={bigrec_hr:.4f}, NDCG@10={bigrec_ndcg:.4f}")

    # --- Evaluate SASRec ---
    logger.info("Evaluating SASRec...")
    # Create standard test loader for SASRec
    sasrec_test_loader = DataLoader(
        dm.test_dataset,
        batch_size=cfg.student.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=student_collator
    )
    
    # Use Trainer to test
    sasrec_trainer_tester = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False, # Don't log to file for this quick check
        enable_checkpointing=False
    )
    sasrec_results = sasrec_trainer_tester.test(sasrec_trainer, dataloaders=sasrec_test_loader)[0]
    
    # --- Evaluate Ensemble ---
    logger.info("Evaluating Ensemble...")
    ensemble_test_dataset = EnsembleDataset(dm.test_dataset, test_bigrec_embs)
    ensemble_test_loader = DataLoader(
        ensemble_test_dataset,
        batch_size=cfg.student.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=ensemble_collator
    )
    
    ensemble_results = trainer.test(ensemble_model, dataloaders=ensemble_test_loader)[0]
    
    # --- Determine Output Directory ---
    ckpt_path_obj = Path(ckpt_path)
    # Assuming structure: .../exp_name/checkpoints/model.ckpt
    # We want: .../exp_name/post_training/
    if ckpt_path_obj.parent.name == "checkpoints":
        output_dir = ckpt_path_obj.parent.parent / "post_training"
    else:
        # Fallback: just create post_training next to ckpt
        output_dir = ckpt_path_obj.parent / "post_training"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    # --- Save Results ---
    import json
    results = {
        "best_lambda": best_lambda,
        "bigrec": {
            "hr@10": bigrec_hr,
            "ndcg@10": bigrec_ndcg
        },
        "sasrec": {
            "hr@10": sasrec_results.get(f"test_hr@{cfg.eval.metrics_k}"),
            "ndcg@10": sasrec_results.get(f"test_ndcg@{cfg.eval.metrics_k}")
        },
        "ensemble": {
            "hr@10": ensemble_results.get("test_hr@10"),
            "ndcg@10": ensemble_results.get("test_ndcg@10"),
            "alpha_mean": ensemble_results.get("test_alpha_mean")
        }
    }
    
    results_path = output_dir / "post_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved results to {results_path.absolute()}")
    
    # --- Save Alpha Network ---
    alpha_net_path = output_dir / "alpha_network.pt"
    torch.save(ensemble_model.alpha_net.state_dict(), alpha_net_path)
    logger.info(f"Saved AlphaNetwork to {alpha_net_path.absolute()}")

if __name__ == "__main__":
    main()
