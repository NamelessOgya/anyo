import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict, Any
import os

class BigRecModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        learning_rate: float = 3e-4,
        max_source_length: int = 512,
        max_target_length: int = 64,
        item_id_to_name: Dict[int, str] = None,
        metrics_k: int = 10,
        num_beams: int = 4,
        item_embeddings_path: str = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 40,
        warmup_steps: int = 20,
        load_in_8bit: bool = False,
        popularity_path: str = None,
        popularity_lambda: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.item_id_to_name = item_id_to_name
        self.metrics_k = metrics_k
        self.num_beams = num_beams
        self.item_embeddings_path = item_embeddings_path
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.padding_side = "left" # Required for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = 0
            self.tokenizer.pad_token = self.tokenizer.decode(0)
            
        # Load Model
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        model_kwargs = {"torch_dtype": torch_dtype}
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            self.model = self.model.to(torch_dtype)
            
        # Load Item Embeddings if provided (Required for Validation)
        self.item_embeddings = None
        if self.item_embeddings_path and os.path.exists(self.item_embeddings_path):
            print(f"Loading item embeddings from {self.item_embeddings_path}...")
            loaded_data = torch.load(self.item_embeddings_path)
            if isinstance(loaded_data, dict):
                max_id = max(loaded_data.keys())
                emb_dim = list(loaded_data.values())[0].shape[0]
                self.item_embeddings = torch.zeros(max_id + 1, emb_dim)
                for k, v in loaded_data.items():
                    self.item_embeddings[k] = v
            else:
                self.item_embeddings = loaded_data
                
        # Load Popularity Scores if provided
        self.popularity_scores = None
        self.popularity_lambda = popularity_lambda
        if popularity_path and os.path.exists(popularity_path):
            print(f"Loading popularity scores from {popularity_path}...")
            self.popularity_scores = torch.load(popularity_path)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        # Standard Causal LM Training
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        self._evaluate_step(batch, batch_idx, prefix="test")

    def _evaluate_step(self, batch, batch_idx, prefix="val"):
        # 1. Calculate Loss (Teacher Forcing)
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        
        # 2. Calculate Metrics (Generation + Grounding)
        if self.item_embeddings is not None:
            if self.item_embeddings.device != self.device:
                self.item_embeddings = self.item_embeddings.to(self.device)

            # Generate predictions (Item Names)
            generated_ids = self.model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=self.hparams.max_target_length,
                num_beams=self.num_beams,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                temperature=self.hparams.get("temperature", 0.0),
                top_p=self.hparams.get("top_p", 0.9),
                top_k=self.hparams.get("top_k", 40),
                do_sample=False if self.hparams.get("temperature", 0.0) == 0 else True
            )
            
            # Extract new tokens
            input_len = batch["prompt_input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            
            # Decode to text
            generated_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            generated_texts = [text.strip('"') for text in generated_texts]
            
            # Embed generated text using Base Model (disable LoRA)
            text_inputs = self.tokenizer(
                generated_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.hparams.max_target_length
            ).to(self.device)
            
            with self.model.disable_adapter():
                text_outputs = self.model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask,
                    output_hidden_states=True
                )
                # Last hidden state
                last_hidden = text_outputs.hidden_states[-1]
                pred_embeddings = last_hidden[:, -1, :] # (B, Dim)
                
            # Compute distances (Euclidean)
            dists = torch.cdist(pred_embeddings.float(), self.item_embeddings.float(), p=2) # (B, NumItems)
            
            # Normalize Distances (Min-Max)
            # Map to [0, 1]
            min_dist = dists.min(dim=1, keepdim=True)[0]
            max_dist = dists.max(dim=1, keepdim=True)[0]
            dists = (dists - min_dist) / (max_dist - min_dist + 1e-8)
            
            # Apply Popularity Adjustment
            # D_new = D_old / (Pop ^ gamma)
            if self.popularity_scores is not None and self.popularity_lambda > 0:
                if self.popularity_scores.device != self.device:
                    self.popularity_scores = self.popularity_scores.to(self.device)
                
                # popularity_scores: (NumItems+1,)
                # dists: (B, NumItems+1) - Wait, item_embeddings has padding at 0?
                # Yes, item_embeddings is (MaxID+1, Dim).
                # So dists is (B, MaxID+1).
                
                # We need to handle padding index 0 or ensure alignment.
                # Assuming popularity_scores is also (MaxID+1).
                
                # Add epsilon to avoid division by zero if pop is 0
                pop_factor = (self.popularity_scores + 1.0) ** self.popularity_lambda
                dists = dists / pop_factor.unsqueeze(0) # Broadcast (1, NumItems)
            
            # Rank (Top-K Smallest Distance)
            topk_dists, topk_indices = torch.topk(dists, k=self.metrics_k, dim=1, largest=False)
            
            # Calculate HR and NDCG
            target_ids = batch["next_item"] # (B,)
            
            hits = 0
            ndcg = 0
            batch_size = len(target_ids)
            
            for i in range(batch_size):
                target = target_ids[i].item()
                preds = topk_indices[i].tolist() # List of K item IDs
                
                if target in preds:
                    hits += 1
                    rank = preds.index(target)
                    ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
            
            self.log(f"{prefix}_hr@{self.metrics_k}", hits / batch_size, prog_bar=True)
            self.log(f"{prefix}_ndcg@{self.metrics_k}", ndcg / batch_size, prog_bar=True)
            
            # Log Samples (First batch only)
            if batch_idx == 0:
                print(f"\n--- {prefix.upper()} Samples ---")
                for i in range(min(3, batch_size)):
                    target_id = target_ids[i].item()
                    target_name = self.item_id_to_name[target_id] if self.item_id_to_name and target_id in self.item_id_to_name else f"ID:{target_id}"
                    gen_text = generated_texts[i]
                    
                    # Get Top-1 Prediction Name
                    top1_pred_id = topk_indices[i][0].item()
                    top1_pred_name = self.item_id_to_name[top1_pred_id] if self.item_id_to_name and top1_pred_id in self.item_id_to_name else f"ID:{top1_pred_id}"
                    
                    print(f"Sample {i}:")
                    print(f"  Target: {target_name}")
                    print(f"  Generated: {gen_text}")
                    print(f"  Retrieved Top-1: {top1_pred_name}")
                    print(f"  Distance: {topk_dists[i][0].item():.4f}")
                    print("-" * 20)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        
        from transformers import get_linear_schedule_with_warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
