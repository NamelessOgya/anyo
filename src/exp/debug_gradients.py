import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from src.teacher.factory import create_teacher_model
from src.student.datamodule import SASRecDataModule
from src.teacher.trainer_ilora import iLoRATrainer
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print("--- Starting Gradient Debugging ---")
    
    # Force settings for debugging
    cfg.teacher.model_type = "ilora"
    cfg.teacher.use_flash_attention = False # Disable Flash Attention for CPU debugging
    cfg.teacher.rec_model_checkpoint_path = None # Disable checkpoint loading
    # cfg.teacher.freeze_item_embeddings = False # Ensure this is False (though factory ignores it, good for clarity)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load DataModule
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=2, # Small batch
        max_seq_len=cfg.student.max_seq_len,
        tokenizer=tokenizer,
        limit_data_rows=100
    )
    dm.prepare_data()
    dm.setup()
    
    # Create Model
    model = create_teacher_model(
        cfg,
        tokenizer,
        num_items=dm.num_items,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=dm.mapped_id_to_title,
        padding_item_id=dm.padding_item_id,
        candidate_topk=10
    )
    
    # Create Trainer Module
    trainer_module = iLoRATrainer(
        ilora_model=model,
        num_items=dm.num_items,
        learning_rate=cfg.teacher.learning_rate,
        weight_decay=0.01,
        metrics_k=10,
        item_id_to_name=dm.mapped_id_to_title
    )
    
    # Get a batch
    batch = next(iter(dm.train_dataloader()))
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer_module.to(device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
            
    # Forward and Backward
    print("Running forward pass...")
    loss = trainer_module.training_step(batch, 0)
    print(f"Loss: {loss.item()}")
    
    print("Running backward pass...")
    loss.backward()
    
    # Check Gradients
    print("\n--- Gradient Check ---")
    
    # 1. Input Embeddings
    input_embeddings = model.llm.get_input_embeddings()
    if input_embeddings.weight.grad is None:
        print("ERROR: input_embeddings.weight.grad is None!")
    else:
        grad = input_embeddings.weight.grad
        # Check original vocab (should be 0 due to hook)
        orig_vocab_grad_norm = grad[:model.original_vocab_size].norm().item()
        print(f"Original Vocab Gradient Norm (should be 0): {orig_vocab_grad_norm}")
        
        # Check item embeddings (should be non-zero)
        item_emb_grad_norm = grad[model.original_vocab_size:].norm().item()
        print(f"Item Embeddings Gradient Norm (should be > 0): {item_emb_grad_norm}")
        
        if item_emb_grad_norm == 0:
            print("WARNING: Item embeddings are NOT updating!")
        else:
            print("SUCCESS: Item embeddings are updating.")

    # 2. LoRA Parameters
    print("\nChecking LoRA gradients...")
    lora_grads = []
    for n, p in model.named_parameters():
        if "lora" in n and p.requires_grad:
            if p.grad is not None:
                lora_grads.append(p.grad.norm().item())
            else:
                print(f"WARNING: {n} has no gradient!")
    
    if lora_grads:
        avg_lora_grad = sum(lora_grads) / len(lora_grads)
        print(f"Average LoRA Gradient Norm: {avg_lora_grad}")
    else:
        print("WARNING: No LoRA parameters found or gradients missing.")

    # 3. Gating Network
    print("\nChecking Gating Network gradients...")
    gating_grads = []
    for n, p in model.gating_network.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                gating_grads.append(p.grad.norm().item())
            else:
                 print(f"WARNING: {n} has no gradient!")
    
    if gating_grads:
        avg_gating_grad = sum(gating_grads) / len(gating_grads)
        print(f"Average Gating Network Gradient Norm: {avg_gating_grad}")
    else:
        print("WARNING: No Gating Network parameters found.")

if __name__ == "__main__":
    main()
