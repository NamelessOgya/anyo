import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import logging

from src.data.datamodule import SASRecDataModule

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # 1. Load DataModule to get item names
    dm = SASRecDataModule(conf=cfg.dataset)
    dm.setup()
    item_id_to_name = dm.item_id_to_name
    
    # 2. Load Base Model (No LoRA)
    model_name = cfg.teacher.llm_model_name
    logger.info(f"Loading base model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()
    
    # 3. Compute Embeddings
    item_embeddings = []
    item_ids = sorted(list(item_id_to_name.keys()))
    
    # Batch processing for efficiency
    batch_size = 32
    
    logger.info(f"Computing embeddings for {len(item_ids)} items...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(item_ids), batch_size)):
            batch_ids = item_ids[i:i+batch_size]
            batch_names = [item_id_to_name[iid] for iid in batch_ids]
            
            # Tokenize
            inputs = tokenizer(
                batch_names,
                padding=True,
                truncation=True,
                max_length=cfg.teacher.max_target_length, # Item titles shouldn't be too long
                return_tensors="pt"
            ).to(model.device)
            
            # Forward
            outputs = model(**inputs)
            
            # Extract last hidden state
            # hidden_states is tuple of (layer_0, ..., layer_N)
            # We want the last layer: hidden_states[-1] -> (B, Seq, Dim)
            last_hidden_state = outputs.hidden_states[-1]
            
            # Get the embedding of the last token (EOS or last char)
            # Attention mask helps find the last valid token
            # inputs['attention_mask'] has 1 for valid, 0 for pad.
            # We want the index of the last '1'.
            
            # Note: if padding_side is right (default), last token is at sum(mask) - 1
            # If padding_side is left, last token is at -1 (if not padded at end? wait)
            # LlamaTokenizer usually defaults to right padding unless configured otherwise.
            # Let's rely on attention_mask.sum(1) - 1
            
            last_token_indices = inputs.attention_mask.sum(1) - 1
            
            # Gather embeddings
            # (B, Dim)
            batch_embeddings = last_hidden_state[torch.arange(len(batch_ids)), last_token_indices]
            
            item_embeddings.append(batch_embeddings.cpu())
            
    # Concatenate
    all_embeddings = torch.cat(item_embeddings, dim=0) # (NumItems, Dim)
    
    # Save
    output_path = "item_embeddings.pt"
    torch.save(all_embeddings, output_path)
    logger.info(f"Saved item embeddings to {output_path}. Shape: {all_embeddings.shape}")

if __name__ == "__main__":
    main()
