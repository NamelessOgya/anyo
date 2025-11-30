import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

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

        # Configure Quantization
        # User reported issues with 8-bit loading. Switching to 4-bit (nf4) as a more robust default for memory saving.
        # CRITICAL: Check for GPU. bitsandbytes 4-bit quantization requires GPU.
        quantization_config = None
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("CUDA detected: Enabling 4-bit quantization (NF4).")
        else:
            print("No CUDA detected: Skipping quantization (CPU mode).")

        # Load Model
        # device_map="auto" is generally good for GPU, but on CPU we might want to let PL handle it or just use default.
        # For 4-bit, device_map="auto" is required.
        load_kwargs = {
            "quantization_config": quantization_config,
            **model_kwargs
        }
        if quantization_config is not None:
             load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs
        )
        
        # Prepare model for k-bit training (important for gradient checkpointing and stability)
        # Only needed if quantized
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
        
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
            
        self.model.print_trainable_parameters()
        
        # Load Item Embeddings if provided
        self.item_embeddings = None
        if self.item_embeddings_path and os.path.exists(self.item_embeddings_path):
            print(f"Loading item embeddings from {self.item_embeddings_path}...")
            self.item_embeddings = torch.load(self.item_embeddings_path)
            # Move to device later or keep on CPU if large? 
            # Ideally move to same device as model during validation.

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        # batch should contain: input_ids, attention_mask, labels
        outputs = self(
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
            # Move embeddings to device if needed
            if self.item_embeddings.device != self.device:
                self.item_embeddings = self.item_embeddings.to(self.device)

            # Generate predictions (Top-1 is enough for grounding based on reference)
            # But we can generate K beams and pick the best one?
            # Reference uses num_beams=4 and takes the first one.
            # Generate predictions (Top-1 is enough for grounding based on reference)
            # But we can generate K beams and pick the best one?
            # Reference uses num_beams=4 and takes the first one.
            generated_ids = self.model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_new_tokens=self.hparams.max_target_length,
                num_beams=self.num_beams,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
                temperature=self.hparams.get("temperature", 0.0), # Reference uses 0
                top_p=self.hparams.get("top_p", 0.9), # Reference uses 0.9
                top_k=self.hparams.get("top_k", 40), # Reference uses 40
                do_sample=False if self.hparams.get("temperature", 0.0) == 0 else True
            )
            
            # Extract new tokens
            input_len = batch["prompt_input_ids"].shape[1]
            new_tokens = generated_ids[:, input_len:]
            
            # Decode to text
            generated_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            # Strip quotes to match reference evaluation logic (Grounding uses unquoted embeddings)
            generated_texts = [text.strip('"') for text in generated_texts]
            
            # Embed generated text using Base Model (disable LoRA)
            # We need to tokenize the generated text first
            # Note: We should use the same tokenizer
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
                # Last hidden state of last token
                # Assuming left padding for generation, but here we just padded normally (right padding default?)
                # Tokenizer default is usually right padding unless set otherwise.
                # In compute_item_embeddings, we set padding_side="left".
                # Here we should check.
                # If right padding, last token is at index (length - 1).
                # Let's use attention_mask to find last real token.
                # With left padding (padding_side="left"), the last token is always at the end of the sequence.
                # So we can just take the last hidden state.
                last_hidden = text_outputs.hidden_states[-1]
                pred_embeddings = last_hidden[:, -1, :] # (B, Dim)
                
            # Compute distances
            # (B, Dim) vs (NumItems, Dim)
            # Euclidean distance
            dists = torch.cdist(pred_embeddings.float(), self.item_embeddings.float(), p=2) # (B, NumItems)
            
            # Rank
            # We want Top-K items with smallest distance
            # dists is (B, NumItems)
            # Get indices of Top-K smallest
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
            
            val_hr = hits / batch_size
            val_ndcg = ndcg / batch_size
            
            self.log(f"{prefix}_hr@{self.metrics_k}", val_hr, prog_bar=True)
            self.log(f"{prefix}_ndcg@{self.metrics_k}", val_ndcg, prog_bar=True)
            
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
