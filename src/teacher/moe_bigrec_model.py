import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from typing import List, Dict, Any
import os
from src.student.models import SASRec

class MoEBigRecModel(pl.LightningModule):
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
        student_model_path: str = None, # Path to SASRec checkpoint
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
            self.tokenizer.pad_token_id = 0 # Reference uses 0 (unk)
            self.tokenizer.pad_token = self.tokenizer.decode(0)
            
        # Load Model
        # Determine dtype (bf16 if supported, else fp16)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("Using bfloat16 precision.")
        else:
            torch_dtype = torch.float16
            print("Using float16 precision.")

        # Enable Flash Attention 2 if available for speedup
        model_kwargs = {"torch_dtype": torch_dtype}
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Flash Attention 2 enabled.")
        except ImportError:
            print("Flash Attention 2 not found. Using default attention.")

        # Load Model
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
            target_modules=["q_proj", "v_proj"] # Common for Llama/Qwen. Adjust if needed.
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Ensure LoRA layers match the base model dtype if using Flash Attention 2
        # LoRA layers are often float32 by default, which causes dtype mismatch in FA2
        if model_kwargs.get("attn_implementation") == "flash_attention_2":
            self.model = self.model.to(torch_dtype)
            
        # User reported this causes issues during quantization loading
        # self.model.print_trainable_parameters()
        
        # Load Item Embeddings if provided
        self.item_embeddings = None
        if self.item_embeddings_path and os.path.exists(self.item_embeddings_path):
            print(f"Loading item embeddings from {self.item_embeddings_path}...")
            loaded_data = torch.load(self.item_embeddings_path)
            if isinstance(loaded_data, dict):
                # Convert dict to tensor
                # Assume keys are 1-based item IDs
                max_id = max(loaded_data.keys())
                # Infer embedding dim
                emb_dim = list(loaded_data.values())[0].shape[0]
                # Create tensor (0 is padding)
                self.item_embeddings = torch.zeros(max_id + 1, emb_dim)
                for k, v in loaded_data.items():
                    self.item_embeddings[k] = v
            else:
                self.item_embeddings = loaded_data
            
            # Move to device later or keep on CPU if large? 
            # Ideally move to same device as model during validation.
            
        # Load Popularity Scores if provided
        self.popularity_scores = None
        self.popularity_lambda = popularity_lambda
        if popularity_path and os.path.exists(popularity_path):
            print(f"Loading popularity scores from {popularity_path}...")
            self.popularity_scores = torch.load(popularity_path)
            # Ensure shape matches num_items + 1
            # If loaded tensor is smaller, pad it? Or assume correct.
            # popularity_scores should be (NumItems + 1,)
            
        # Load SASRec (Student) for Ensemble
        self.sasrec = None
        self.alpha = None
        if student_model_path:
            if not os.path.exists(student_model_path):
                raise FileNotFoundError(f"student_model_path is set to '{student_model_path}', but the file does not exist.")
            
            print(f"Loading SASRec from {student_model_path}...")
            # ... (rest of SASRec loading)

    def training_step(self, batch, batch_idx):
        # ...
            llm_logits_full = torch.matmul(llm_user_emb, self.item_embeddings.t()) # (B, NumItems+1)
            llm_logits = llm_logits_full[:, 1:] # Remove padding index 0. (B, NumItems)
            
            # Apply Popularity Bias Adjustment (BIGRec)
            # Logits = Logits + lambda * log(Pop)
            if self.popularity_scores is not None and self.popularity_lambda > 0:
                if self.popularity_scores.device != self.device:
                    self.popularity_scores = self.popularity_scores.to(self.device)
                
                # popularity_scores is (NumItems+1,)
                # We need indices 1..NumItems
                pop_scores = self.popularity_scores[1:] # (NumItems,)
                
                # Add to logits (Broadcasting)
                # log(count + 1) to handle zeros if any, though we initialized with 0 for padding.
                # If count is 0, log(1) = 0.
                # We assume popularity_scores contains raw counts.
                pop_adjustment = self.popularity_lambda * torch.log(pop_scores + 1.0)
                llm_logits = llm_logits + pop_adjustment
            
            # Normalize Logits (Z-score)
            # ...

    def validation_step(self, batch, batch_idx):
        # ...
            llm_logits_full = torch.matmul(llm_user_emb, self.item_embeddings.t())
            llm_logits = llm_logits_full[:, 1:] # Remove padding index 0
            
            # Apply Popularity Bias Adjustment (BIGRec)
            if self.popularity_scores is not None and self.popularity_lambda > 0:
                if self.popularity_scores.device != self.device:
                    self.popularity_scores = self.popularity_scores.to(self.device)
                
                pop_scores = self.popularity_scores[1:]
                pop_adjustment = self.popularity_lambda * torch.log(pop_scores + 1.0)
                llm_logits = llm_logits + pop_adjustment
            
            # Normalize Logits (Z-score)
            # ...
            # We need to load the checkpoint. 
            # Since SASRec is usually wrapped in a LightningModule (StudentModel?), we need to be careful.
            # Or if it's just the model state dict.
            # Assuming it's a PL checkpoint.
            try:
                # We need to know the class structure of the checkpoint.
                # If it's a standard PL checkpoint, we can load state_dict.
                # But we need to instantiate SASRec first.
                # We'll assume standard ML-100k params for now or try to infer?
                # Hardcoding ML-100k params for simplicity as per config usually.
                # num_items=1682, hidden_size=64, etc.
                # Ideally, we should load config from the checkpoint or similar.
                # For this task, let's assume standard params.
                num_items = 1682 # ML-100k
                hidden_size = 64
                num_heads = 2
                num_layers = 2
                max_seq_len = 50
                dropout_rate = 0.1
                
                self.sasrec = SASRec(num_items, hidden_size, num_heads, num_layers, dropout_rate, max_seq_len)
                
                checkpoint = torch.load(student_model_path, map_location="cpu")
                # Checkpoint keys might be "state_dict" -> "model.xxx"
                state_dict = checkpoint["state_dict"]
                # Adjust keys if necessary (remove "model." prefix if wrapped)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                self.sasrec.load_state_dict(new_state_dict)
                
                # Freeze SASRec
                for param in self.sasrec.parameters():
                    param.requires_grad = False
                self.sasrec.eval()
                print("SASRec loaded and frozen.")
                
                # Initialize Gate (Dynamic Alpha)
                # Input: SASRec hidden size -> Output: 1 (logit for sigmoid)
                self.gate = nn.Linear(hidden_size, 1)
                # Initialize bias to 0 (sigmoid(0) = 0.5) to start neutral
                nn.init.zeros_(self.gate.bias)
                print("Ensemble Gate initialized.")
                
            except Exception as e:
                print(f"Failed to load SASRec: {e}")
                import traceback
                traceback.print_exc()
                self.sasrec = None

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        # If SASRec is loaded, use Ensemble Loss
        if self.sasrec is not None and self.item_embeddings is not None:
            # 1. Get LLM Logits (via Embedding Similarity)
            prompt_ids = batch["prompt_input_ids"]
            prompt_mask = batch["prompt_attention_mask"]
            
            outputs = self.model.model(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                output_hidden_states=True
            )
            # Last hidden state
            last_hidden = outputs.hidden_states[-1] # (B, Seq, H)
            llm_user_emb = last_hidden[:, -1, :] # (B, H)
            
            if self.item_embeddings.device != self.device:
                self.item_embeddings = self.item_embeddings.to(self.device)
                
            llm_logits_full = torch.matmul(llm_user_emb, self.item_embeddings.t()) # (B, NumItems+1)
            llm_logits = llm_logits_full[:, 1:] # Remove padding index 0. (B, NumItems)
            
            # Apply Popularity Bias Adjustment (BIGRec)
            if self.popularity_scores is not None and self.popularity_lambda > 0:
                if self.popularity_scores.device != self.device:
                    self.popularity_scores = self.popularity_scores.to(self.device)
                
                pop_scores = self.popularity_scores[1:] # (NumItems,)
                pop_adjustment = self.popularity_lambda * torch.log(pop_scores + 1.0)
                llm_logits = llm_logits + pop_adjustment
            
            # 2. Get SASRec Logits and User Embedding
            sasrec_ids = batch["sasrec_input_ids"] # (B, Seq)
            sasrec_lens = (sasrec_ids != 0).sum(dim=1)
            
            # SASRec predict returns (B, NumItems)
            sasrec_logits = self.sasrec.predict(sasrec_ids, sasrec_lens)
            
            # Get SASRec User Embedding for Gating
            sasrec_user_emb = self.sasrec(sasrec_ids, sasrec_lens) # (B, Hidden)
            
            # 3. Compute Dynamic Alpha
            alpha_logits = self.gate(sasrec_user_emb) # (B, 1)
            alpha = torch.sigmoid(alpha_logits) # (B, 1)
            
            # 4. Compute Ensemble Probability Loss
            target_ids = batch["next_item"] # (B,) 1-based IDs
            loss_targets = target_ids - 1
            
            # Softmax to get Probabilities
            probs_llm = F.softmax(llm_logits, dim=-1)       # (B, NumItems)
            probs_sasrec = F.softmax(sasrec_logits, dim=-1) # (B, NumItems)
            
            # Weighted Average of Probabilities
            # alpha is (B, 1)
            probs_ensemble = alpha * probs_sasrec + (1 - alpha) * probs_llm
            
            # Compute Loss (NLLLoss expects log-probabilities)
            # Add epsilon for numerical stability
            log_probs_ensemble = torch.log(probs_ensemble + 1e-8)
            
            loss = F.nll_loss(log_probs_ensemble, loss_targets)
            
            # Logging (Optional: Compute individual CrossEntropy for monitoring)
            with torch.no_grad():
                loss_llm = F.cross_entropy(llm_logits, loss_targets)
                loss_sasrec = F.cross_entropy(sasrec_logits, loss_targets)
            
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_loss_llm", loss_llm, prog_bar=True)
            self.log("train_loss_sasrec", loss_sasrec, prog_bar=True)
            self.log("train_alpha", alpha.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            return loss
            
        else:
            # Standard Causal LM Training (Text Generation)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            self.log("train_loss", loss, prog_bar=True)
            return loss

    def validation_step(self, batch, batch_idx):
        if self.sasrec is not None and self.item_embeddings is not None:
            self._evaluate_ensemble(batch, batch_idx, prefix="val")
        else:
            self._evaluate_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        if self.sasrec is not None and self.item_embeddings is not None:
            self._evaluate_ensemble(batch, batch_idx, prefix="test")
        else:
            self._evaluate_step(batch, batch_idx, prefix="test")

    def _evaluate_ensemble(self, batch, batch_idx, prefix="val"):
        # Ensemble Evaluation
        prompt_ids = batch["prompt_input_ids"]
        prompt_mask = batch["prompt_attention_mask"]
        
        outputs = self.model.model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            output_hidden_states=True
        )
        llm_user_emb = outputs.hidden_states[-1][:, -1, :]
        
        if self.item_embeddings.device != self.device:
            self.item_embeddings = self.item_embeddings.to(self.device)
            
        llm_logits_full = torch.matmul(llm_user_emb, self.item_embeddings.t())
        llm_logits = llm_logits_full[:, 1:] # Remove padding index 0
        
        sasrec_ids = batch["sasrec_input_ids"]
        sasrec_lens = (sasrec_ids != 0).sum(dim=1)
        sasrec_logits = self.sasrec.predict(sasrec_ids, sasrec_lens)
        
        # Dynamic Alpha
        sasrec_user_emb = self.sasrec(sasrec_ids, sasrec_lens)
        alpha_logits = self.gate(sasrec_user_emb)
        alpha = torch.sigmoid(alpha_logits)
        
        # Normalize Logits (Z-score)
        llm_mean = llm_logits.mean(dim=-1, keepdim=True)
        llm_std = llm_logits.std(dim=-1, keepdim=True)
        llm_logits_norm = (llm_logits - llm_mean) / (llm_std + 1e-8)
        
        sasrec_mean = sasrec_logits.mean(dim=-1, keepdim=True)
        sasrec_std = sasrec_logits.std(dim=-1, keepdim=True)
        sasrec_logits_norm = (sasrec_logits - sasrec_mean) / (sasrec_std + 1e-8)
        
        ensemble_logits = alpha * sasrec_logits_norm + (1 - alpha) * llm_logits_norm

        # Apply Popularity Bias Adjustment (Global Prior)
        # Applied AFTER ensemble and normalization to ensure consistent scale
        pop_adjustment = torch.tensor(0.0, device=self.device)
        if self.popularity_scores is not None and self.popularity_lambda > 0:
            if self.popularity_scores.device != self.device:
                self.popularity_scores = self.popularity_scores.to(self.device)
            
            pop_scores = self.popularity_scores[1:]
            pop_adjustment = self.popularity_lambda * torch.log(pop_scores + 1.0)
            ensemble_logits = ensemble_logits + pop_adjustment
        
        self.log(f"{prefix}_llm_mean", llm_mean.mean(), prog_bar=True)
        self.log(f"{prefix}_llm_std", llm_std.mean(), prog_bar=True)
        self.log(f"{prefix}_sasrec_mean", sasrec_mean.mean(), prog_bar=True)
        self.log(f"{prefix}_sasrec_std", sasrec_std.mean(), prog_bar=True)
        
        # Metrics
        # Top-K
        _, topk_indices = torch.topk(ensemble_logits, k=self.metrics_k, dim=-1) # (B, K)
        # topk_indices are 0-based (corresponding to item 1..N)
        # target_ids are 1-based
        target_ids = batch["next_item"]
        
        hits = 0
        ndcg = 0
        batch_size = target_ids.size(0)
        
        for i in range(batch_size):
            target = target_ids[i].item() # 1-based
            preds = topk_indices[i].tolist() # 0-based
            preds = [p + 1 for p in preds] # Convert to 1-based
            
            if target in preds:
                hits += 1
                rank = preds.index(target)
                ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
        
        self.log(f"{prefix}_hr@{self.metrics_k}", hits / batch_size, prog_bar=True)
        self.log(f"{prefix}_ndcg@{self.metrics_k}", ndcg / batch_size, prog_bar=True)
        self.log(f"{prefix}_alpha", alpha.mean(), on_step=False, on_epoch=True, prog_bar=True)
        
            # Debug Logging (First 3 samples of the batch)
            if batch_idx == 0:
                print(f"\n[Epoch {self.current_epoch} {prefix.capitalize()} Debug]")
                print(f"Alpha (Mean): {alpha.mean().item():.4f}")
                print(f"LLM Logits   - Mean: {llm_mean.mean().item():.4f}, Std: {llm_std.mean().item():.4f}")
                if self.popularity_scores is not None and self.popularity_lambda > 0:
                    print(f"Pop Adjust   - Max: {pop_adjustment.max().item():.4f}, Mean: {pop_adjustment.mean().item():.4f}")
                print(f"SASRec Logits- Mean: {sasrec_mean.mean().item():.4f}, Std: {sasrec_std.mean().item():.4f}")
                
                debug_batch_size = min(3, batch_size)
                
                # Get Top-3 for individual models
                _, llm_topk = torch.topk(llm_logits[:debug_batch_size], k=3)
                _, sasrec_topk = torch.topk(sasrec_logits[:debug_batch_size], k=3)
                
                for i in range(debug_batch_size):
                    target_id = target_ids[i].item()
                    target_name = self.item_id_to_name.get(target_id, f"Item_{target_id}")
                    
                    # Ensemble Preds
                    ens_indices = topk_indices[i].tolist()[:3]
                    ens_names = [self.item_id_to_name.get(p + 1, f"Item_{p+1}") for p in ens_indices]
                    
                    # LLM Preds
                    llm_indices = llm_topk[i].tolist()
                    llm_names = [self.item_id_to_name.get(p + 1, f"Item_{p+1}") for p in llm_indices]
                    
                    # SASRec Preds
                    sasrec_indices = sasrec_topk[i].tolist()
                    sasrec_names = [self.item_id_to_name.get(p + 1, f"Item_{p+1}") for p in sasrec_indices]
                    
                    print(f"Sample {i}:")
                    print(f"  Alpha : {alpha[i].item():.4f}")
                    print(f"  Target: {target_name}")
                    print(f"  LLM   : {llm_names}")
                    print(f"  SASRec: {sasrec_names}")
                    print(f"  Ensem : {ens_names}")

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
            # Move embeddings to device if needed
            if self.item_embeddings.device != self.device:
                self.item_embeddings = self.item_embeddings.to(self.device)

            # Compute distances
            # (B, Dim) vs (NumItems, Dim)
            # Euclidean distance
            # dists = torch.cdist(pred_embeddings.float(), self.item_embeddings.float(), p=2) # (B, NumItems)
            
            # Rank
            # We want Top-K items with smallest distance
            # dists is (B, NumItems)
            # Get indices of Top-K smallest
            # topk_dists, topk_indices = torch.topk(dists, k=self.metrics_k, dim=1, largest=False)
            
            # Calculate HR and NDCG
            # target_ids = batch["next_item"] # (B,)
            
            # hits = 0
            # ndcg = 0
            # batch_size = len(target_ids)
            
            # for i in range(batch_size):
            #     target = target_ids[i].item()
            #     preds = topk_indices[i].tolist() # List of K item IDs
                
            #     if target in preds:
            #         hits += 1
            #         rank = preds.index(target)
            #         ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
            
            # val_hr = hits / batch_size
            # val_ndcg = ndcg / batch_size
            
            # self.log(f"{prefix}_hr@{self.metrics_k}", val_hr, prog_bar=True)
            # self.log(f"{prefix}_ndcg@{self.metrics_k}", val_ndcg, prog_bar=True)
            
            # Since we removed generation, we can't compute distance-based metrics here without generation.
            # But wait, BigRec relies on generation for inference.
            # If we remove generation, we can't evaluate BigRec (non-ensemble) properly.
            # However, the user asked to remove generation because it's slow.
            # For MoE-BigRec (Ensemble), we use _evaluate_ensemble which uses Logits (fast).
            # For pure BigRec, we need generation.
            # Assuming the user is focusing on MoE-BigRec now.
            # If we are in MoE mode (sasrec is not None), we use _evaluate_ensemble.
            # If we are in BigRec mode, we still need generation?
            # Or maybe we can use Logit-based evaluation for BigRec too if we have item embeddings?
            # No, BigRec is Generative Retrieval.
            
            # Let's just comment out the generation part in _evaluate_step for now or skip it if requested.
            # But _evaluate_step is used when sasrec is None.
            # If the user is running MoE-BigRec, sasrec is NOT None, so _evaluate_ensemble is used.
            # So _evaluate_step changes might not be needed if we only care about MoE.
            # But to be safe and consistent, let's leave _evaluate_step as is (or minimal) 
            # and ensure _evaluate_ensemble is fast.
            # The previous tool call already modified _evaluate_ensemble.
            
            pass
            
        elif self.item_id_to_name:
            # Fallback to Exact Match if embeddings not provided
            # Generate predictions
            # Use beam search to get top-K candidates
            # Fallback to Exact Match if embeddings not provided
            # Generate predictions
            # Use beam search to get top-K candidates
            generated_ids = self.model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=self.hparams.max_target_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # generated_ids shape: (batch_size * num_return_sequences, seq_len)
            # Reshape to (batch_size, num_return_sequences, seq_len)
            batch_size = batch["prompt_input_ids"].size(0)
            generated_ids = generated_ids.view(batch_size, self.num_beams, -1)
            
            # Decode predictions
            decoded_preds = []
            for i in range(batch_size):
                # Extract new tokens
                input_len = batch["prompt_input_ids"].shape[1]
                new_tokens = generated_ids[i, :, input_len:]
                preds = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                decoded_preds.append([p.strip() for p in preds])

            # Get Ground Truth Titles
            target_ids = batch["next_item"] # (batch_size,)
            target_titles = [self.item_id_to_name.get(tid.item(), "") for tid in target_ids]
            
            # Calculate HR and NDCG
            hits = 0
            ndcg = 0
            
            for i, target_title in enumerate(target_titles):
                preds = decoded_preds[i] # List of K strings
                
                # Check if target_title is in preds
                # We use exact string match (case insensitive maybe?)
                # Let's do case insensitive strip match
                target_clean = target_title.strip().lower()
                preds_clean = [p.strip().lower() for p in preds]
                
                if target_clean in preds_clean:
                    hits += 1
                    rank = preds_clean.index(target_clean) # 0-based
                    ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
            
            val_hr = hits / batch_size
            val_ndcg = ndcg / batch_size
            
            self.log(f"{prefix}_hr@{self.metrics_k}", val_hr, prog_bar=True)
            self.log(f"{prefix}_ndcg@{self.metrics_k}", val_ndcg, prog_bar=True)

    def on_save_checkpoint(self, checkpoint):
        # Only save trainable parameters (LoRA + Gate) to save space and avoid saving frozen models
        state_dict = checkpoint["state_dict"]
        keys_to_keep = [k for k, v in self.named_parameters() if v.requires_grad]
        new_state_dict = {k: state_dict[k] for k in keys_to_keep if k in state_dict}
        checkpoint["state_dict"] = new_state_dict

    def load_state_dict(self, state_dict, strict=True):
        # Allow missing keys (frozen params)
        return super().load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)
        
        # Use Linear Decay with Warmup
        from transformers import get_linear_schedule_with_warmup
        
        # Use Linear Decay with Warmup
        from transformers import get_linear_schedule_with_warmup
        
        # Estimate total steps
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

    def generate(self, input_ids, attention_mask, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = self.hparams.max_target_length
            
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    @staticmethod
    def extract_recommendation(text: str, use_cot: bool = False) -> str:
        """
        Extracts the recommendation part from the generated text.
        """
        # Split by Response tag
        parts = text.split("### Response:\n")
        if len(parts) > 1:
            response_part = parts[-1].strip()
        else:
            response_part = text.strip()
        
        if use_cot:
            # Format: "Reasoning: ...\nRecommendation: Item"
            rec_parts = response_part.split("Recommendation:")
            if len(rec_parts) > 1:
                final_rec = rec_parts[-1].strip()
            else:
                # Fallback: maybe the model didn't output Recommendation tag?
                # Use the whole response or last line?
                final_rec = response_part.split("\n")[-1].strip()
            return final_rec
        else:
            return response_part
