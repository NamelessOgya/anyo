import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
import os

from src.teacher.bigrec_model import BigRecModel
from src.data.datamodule import SASRecDataModule
from src.data.collators import BigRecCollator

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # 1. Load Data
    dm = SASRecDataModule(conf=cfg.dataset)
    dm.setup()
    item_id_to_name = dm.item_id_to_name
    
    # 2. Load Model (Fine-tuned)
    # Assuming checkpoint path is provided or we load from default output dir
    # For inference, we usually need a specific checkpoint.
    # Let's assume user provides it via cfg.teacher.checkpoint_path or similar.
    # If not, we load base model + LoRA adapters if saved separately.
    # BigRecModel loads base model in __init__.
    
    model = BigRecModel(
        model_name_or_path=cfg.teacher.llm_model_name,
        lora_r=cfg.teacher.lora_r,
        lora_alpha=cfg.teacher.lora_alpha,
        lora_dropout=cfg.teacher.lora_dropout,
        max_source_length=cfg.teacher.max_source_length,
        max_target_length=cfg.teacher.max_target_length
    )
    
    # Load LoRA weights if provided
    if hasattr(cfg.teacher, "checkpoint_path") and cfg.teacher.checkpoint_path:
        logger.info(f"Loading checkpoint from {cfg.teacher.checkpoint_path}")
        # If it's a PL checkpoint, we load state dict.
        # If it's a PEFT adapter path, we use load_adapter.
        # Since BigRecModel is a PL module, we use load_from_checkpoint.
        model = BigRecModel.load_from_checkpoint(cfg.teacher.checkpoint_path)
    
    model.eval()
    model.cuda()
    
    # 3. Load Item Embeddings
    embedding_path = "item_embeddings.pt"
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"{embedding_path} not found. Run compute_item_embeddings.py first.")
    
    item_embeddings = torch.load(embedding_path).cuda() # (NumItems, Dim)
    
    # 4. Inference Loop
    collator = BigRecCollator(
        tokenizer=model.tokenizer,
        item_id_to_name=item_id_to_name,
        max_source_length=cfg.teacher.max_source_length,
        max_target_length=cfg.teacher.max_target_length,
        use_cot=cfg.teacher.get("use_cot", False)
    )
    
    test_loader = DataLoader(
        dm.test_dataset,
        batch_size=cfg.teacher.batch_size, # Use smaller batch for generation
        shuffle=False,
        num_workers=dm.conf.num_workers,
        collate_fn=collator
    )
    
    results = []
    
    logger.info("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Use prompt_input_ids for generation
            input_ids = batch["prompt_input_ids"].cuda()
            attention_mask = batch["prompt_attention_mask"].cuda()
            
            # Generate Text
            generated_ids = model.generate(input_ids, attention_mask)
            
            generated_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Extract Response part
            responses = []
            use_cot = cfg.teacher.get("use_cot", False)
            for text in generated_text:
                final_rec = BigRecModel.extract_recommendation(text, use_cot=use_cot)
                responses.append(final_rec)
            
            # Step B: Get Embedding of Generated Text (for Grounding)
            # Tokenize responses (the extracted item titles)
            resp_inputs = model.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=cfg.teacher.max_target_length,
                return_tensors="pt"
            ).to(model.device)
            
            resp_outputs = model.model(
                input_ids=resp_inputs.input_ids,
                attention_mask=resp_inputs.attention_mask,
                output_hidden_states=True
            )
            
            # Last hidden state of last token
            last_hidden = resp_outputs.hidden_states[-1]
            last_token_idx = resp_inputs.attention_mask.sum(1) - 1
            pred_embeddings = last_hidden[torch.arange(len(responses)), last_token_idx] # (B, Dim)
            
            # Step C: Grounding (Euclidean Distance)
            dists = torch.cdist(pred_embeddings, item_embeddings, p=2) # (B, NumItems)
            
            # Top-K
            topk_dists, topk_indices = torch.topk(dists, k=10, dim=1, largest=False)
            
            sorted_item_ids = sorted(list(item_id_to_name.keys()))
            
            for b in range(len(responses)):
                pred_ids = [sorted_item_ids[idx] for idx in topk_indices[b].cpu().numpy()]
                results.append({
                    "generated_text": responses[b], # This is the extracted recommendation
                    "full_generated_text": generated_text[b], # Save full text for debugging
                    "pred_item_ids": pred_ids
                })
                
    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Inference completed. Results saved to inference_results.json")

if __name__ == "__main__":
    main()
