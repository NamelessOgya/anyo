import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.teacher.ilora_model import iLoRAModel
from src.teacher.trainer_ilora import iLoRATrainer
from src.teacher.factory import create_teacher_model
from transformers import AutoTokenizer
import os

def test_reverse_distillation_loss():
    # 1. Setup Dummy Config
    cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 2,
            "lora_r": 4,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "hidden_size": 32,
            "dropout_rate": 0.0,
            "rec_model_checkpoint_path": "dummy_ckpt_rd.pth",
            "use_flash_attention": False,
            "use_qlora": False,
            "use_gradient_checkpointing": False,
            "use_torch_compile": False,
            "distill_lambda": 1.0 # Set to 1.0 for easy calculation check
        },
        "student": {
            "hidden_size": 32,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.0,
            "max_seq_len": 20,
        },
        "train": {
            "precision": "32",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "val_check_interval": 1.0,
            "log_every_n_steps": 1,
            "accumulate_grad_batches": 1
        },
        "eval": {
            "metrics_k": [10]
        }
    })

    # 2. Create Dummy Checkpoint
    ckpt_path = os.path.abspath("dummy_ckpt_rd.pth")
    cfg.teacher.rec_model_checkpoint_path = ckpt_path
    
    from src.student.models import SASRec
    rec_model = SASRec(num_items=100, hidden_size=32, num_heads=2, num_layers=2, dropout_rate=0.0, max_seq_len=20, padding_item_id=0)
    
    # Wrap in 'model.' prefix as expected by factory
    save_dict = {'state_dict': {f'model.{k}': v for k, v in rec_model.state_dict().items()}}
    torch.save(save_dict, ckpt_path)

    # 3. Create Teacher Model
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    
    try:
        model = create_teacher_model(
            cfg,
            llm_tokenizer=tokenizer,
            num_items=100,
            max_seq_len=20,
            item_id_to_name={i: str(i) for i in range(101)},
            padding_item_id=0,
            candidate_topk=5
        )

        # 4. Verify Initial State
        assert hasattr(model, "student_item_embeddings"), "Model should have student_item_embeddings buffer"
        assert torch.allclose(model.rec_model.item_embeddings.weight, model.student_item_embeddings), "Initial embeddings should match"
        assert model.student_item_embeddings.requires_grad == False, "Student embeddings should be frozen"
        assert model.rec_model.item_embeddings.weight.requires_grad == True, "Item embeddings should be trainable"

        # 5. Initialize Trainer
        trainer_model = iLoRATrainer(
            ilora_model=model,
            num_items=100,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            metrics_k=cfg.eval.metrics_k,
            item_id_to_name={i: str(i) for i in range(101)},
            distill_lambda=cfg.teacher.distill_lambda
        )

        # 6. Simulate Training Step
        # Manually modify item embeddings to simulate divergence
        with torch.no_grad():
            model.rec_model.item_embeddings.weight[1] += 1.0 # Modify item 1
        
        # Check if loss is calculated correctly
        # MSE for item 1: (1.0)^2 = 1.0. Average over all items (101 items).
        # MSE = 1.0 / 101 / 32 (embedding dim)? No, MSE is mean over all elements.
        # Elements changed: 32 elements of item 1 changed by 1.0.
        # Total elements: 101 * 32.
        # MSE = (32 * 1.0^2) / (101 * 32) = 1/101 ~= 0.0099
        
        expected_mse = 1.0 / 101
        
        # Create dummy batch
        seq = torch.zeros(2, 20, dtype=torch.long)
        seq[:, -5:] = torch.randint(1, 100, (2, 5))
        batch = {
            "seq": seq,
            "len_seq": torch.tensor([5, 5]),
            "input_ids": torch.randint(0, 1000, (2, 10)),
            "attention_mask": torch.ones(2, 10),
            "next_item": torch.tensor([1, 2])
        }
        
        # Forward pass
        loss = trainer_model.training_step(batch, 0)
        
        # We can't easily check the exact loss value because it includes CrossEntropy.
        # But we can check if reg_loss component is added.
        # Let's check if gradients are generated for item_embeddings
        loss.backward()
        
        assert model.rec_model.item_embeddings.weight.grad is not None, "Gradients should be computed for item embeddings"
        assert torch.norm(model.rec_model.item_embeddings.weight.grad) > 0, "Gradients should be non-zero"
        
        # Verify student_item_embeddings did not change
        # We need to reload the checkpoint to compare with original original, 
        # but we can just check if it's still equal to the "original" we modified from?
        # No, student_item_embeddings should remain as the loaded checkpoint values.
        # We manually modified rec_model.item_embeddings, so they should differ now.
        assert not torch.allclose(model.rec_model.item_embeddings.weight, model.student_item_embeddings), "Embeddings should differ after manual modification"
        
        # Check if student_item_embeddings is still same as initial checkpoint
        # Load checkpoint again to verify
        ckpt = torch.load(ckpt_path)
        original_weight = ckpt['state_dict']['model.item_embeddings.weight']
        assert torch.allclose(model.student_item_embeddings, original_weight), "Student embeddings buffer should not change"

    finally:
        # Cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
