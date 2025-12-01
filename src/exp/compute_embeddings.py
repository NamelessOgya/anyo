import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from pathlib import Path
from src.student.datamodule import SASRecDataModule
from src.core.seed import set_seed

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed)
    
    # 1. Load DataModule to get item names
    print("Loading DataModule...")
    # Initialize tokenizer for DataModule (needed for some logic, though we use LLM tokenizer for embeddings)
    # Just use a dummy or load the real one
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name, use_fast=False)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token_id = 0
        llm_tokenizer.pad_token = llm_tokenizer.decode(0)
        
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
    
    item_id_to_name = dm.mapped_id_to_title
    num_items = dm.num_items
    print(f"Loaded {num_items} items.")
    
    # 2. Load LLM
    print(f"Loading LLM: {cfg.teacher.llm_model_name}")
    model_kwargs = {}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float16
        
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher.llm_model_name,
        **model_kwargs
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # 3. Compute Embeddings
    print("Computing Item Embeddings...")
    input_embeddings = model.get_input_embeddings()
    
    # Output Tensor: (NumItems + 1, HiddenSize)
    # Index 0 is padding
    emb_dim = input_embeddings.weight.shape[1]
    item_embeddings = torch.zeros(num_items + 1, emb_dim)
    
    count = 0
    log_interval = 100
    
    with torch.no_grad():
        for item_id, item_name in item_id_to_name.items():
            # Tokenize name
            # add_special_tokens=False to get pure word embeddings
            tokenized = llm_tokenizer(item_name, return_tensors="pt", add_special_tokens=False).to(device)
            input_ids = tokenized.input_ids
            
            if input_ids.size(1) > 0:
                # Get embeddings
                embs = input_embeddings(input_ids) # (1, Seq, Hidden)
                # Mean pooling
                mean_emb = embs.mean(dim=1).squeeze(0) # (Hidden,)
                
                # Store in CPU tensor
                item_embeddings[item_id] = mean_emb.cpu()
            
            count += 1
            if count % log_interval == 0:
                print(f"Processed {count}/{num_items} items...")
                
    # 4. Save
    output_path = cfg.teacher.get("item_embeddings_path", "data/ml-100k/item_embeddings.pt")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving embeddings to {output_path}...")
    torch.save(item_embeddings, output_path)
    print("Done.")

if __name__ == "__main__":
    main()
