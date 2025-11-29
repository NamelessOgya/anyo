import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List, Dict, Any

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
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            # device_map="auto" # Removed to allow PL to handle device placement in DDP
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
        self.model.print_trainable_parameters()

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
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)

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
