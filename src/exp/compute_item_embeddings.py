import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def compute_embeddings(cfg: DictConfig):
    # 1. Load Data
    data_dir = Path(cfg.dataset.data_dir)
    movies_path = data_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"{movies_path} not found. Run preprocess_data.py first.")
    
    logger.info(f"Loading items from {movies_path}...")
    movies_df = pd.read_csv(movies_path)
    item_titles = movies_df['title'].tolist()
    item_ids = movies_df['item_id'].tolist()
    
    # 2. Load Base Model
    model_name = cfg.teacher.llm_model_name
    logger.info(f"Loading base model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" # Important for batch generation/embedding

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # 3. Compute Embeddings
    logger.info("Computing item embeddings...")
    batch_size = 16 # Adjust based on GPU memory
    
    max_item_id = max(item_ids)
    all_embeddings = torch.zeros((max_item_id + 1, model.config.hidden_size), dtype=torch.float16)
    
    # Process in batches
    for i in tqdm(range(0, len(item_titles), batch_size)):
        batch_titles = item_titles[i:i+batch_size]
        batch_ids = item_ids[i:i+batch_size]
        
        inputs = tokenizer(
            batch_titles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.teacher.max_target_length
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            
        # Get last hidden state of the last token
        last_hidden_state = outputs.hidden_states[-1] # (B, Seq, Dim)
        
        # With left padding (padding_side="left"), the last token is always at the end of the sequence.
        # So we can just take the last hidden state.
        batch_embeddings = last_hidden_state[:, -1, :] # (B, Dim)
        
        for j, item_id in enumerate(batch_ids):
            all_embeddings[item_id] = batch_embeddings[j].cpu()
            
    # 4. Save
    if cfg.teacher.get("item_embeddings_path"):
        output_path = Path(cfg.teacher.item_embeddings_path)
    else:
        output_path = data_dir / "item_embeddings.pt"
        
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving embeddings to {output_path}...")
    torch.save(all_embeddings, output_path)
    logger.info("Done.")
    
    # Cleanup to free memory for training
    del model
    del tokenizer
    torch.cuda.empty_cache()

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    compute_embeddings(cfg)


if __name__ == "__main__":
    main()
