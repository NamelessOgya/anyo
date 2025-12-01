import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def compute_and_save_item_embeddings(cfg, llm_tokenizer, item_id_to_name: Dict[int, str], num_items: int):
    """
    Computes item embeddings using the LLM and saves them to disk.
    """
    output_path = cfg.teacher.get("item_embeddings_path", "data/ml-100k/item_embeddings.pt")
    
    # Check if exists and skip if not forced (optional, but user asked to "put it in the process", implying automation)
    # If compute_item_embeddings is True, we force recompute or check existence?
    # Usually "compute_item_embeddings: true" means "ensure they are computed".
    # If they exist, maybe skip? But if parameters changed, we should recompute.
    # Let's assume if this function is called, we want to compute.
    
    if os.path.exists(output_path):
        logger.info(f"Item embeddings already exist at {output_path}. Skipping computation.")
        return

    logger.info(f"Computing item embeddings for {num_items} items...")
    
    # Load LLM
    logger.info(f"Loading LLM: {cfg.teacher.llm_model_name}")
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
    
    input_embeddings = model.get_input_embeddings()
    emb_dim = input_embeddings.weight.shape[1]
    item_embeddings = torch.zeros(num_items + 1, emb_dim)
    
    count = 0
    log_interval = 1000
    
    with torch.no_grad():
        for item_id, item_name in item_id_to_name.items():
            tokenized = llm_tokenizer(item_name, return_tensors="pt", add_special_tokens=False).to(device)
            input_ids = tokenized.input_ids
            
            if input_ids.size(1) > 0:
                embs = input_embeddings(input_ids)
                mean_emb = embs.mean(dim=1).squeeze(0)
                item_embeddings[item_id] = mean_emb.cpu()
            
            count += 1
            if count % log_interval == 0:
                logger.info(f"Processed {count}/{num_items} items...")
                
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    logger.info(f"Saving embeddings to {output_path}...")
    torch.save(item_embeddings, output_path)
    logger.info("Done computing embeddings.")
    
    # Clean up to free memory
    del model
    torch.cuda.empty_cache()
