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

def test_embedding_imitation_methods():
    # 1. Setup Dummy Config Base
    base_cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 2,
            "lora_r": 4,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "hidden_size": 32,
            "dropout_rate": 0.0,
            "rec_model_checkpoint_path": "dummy_ckpt_imit.pth",
            "use_flash_attention": False,
            "use_qlora": False,
            "use_gradient_checkpointing": False,
            "use_torch_compile": False,
            "distill_lambda": 1.0,
            "use_item_embeddings_head": True
        },
        "student": {
            "hidden_size": 32,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.0,
            "max_seq_len": 20,
        },
        "train": {
            "precision": "32"
        }
    })

    # 2. Create Dummy Checkpoint
    ckpt_path = os.path.abspath("dummy_ckpt_imit.pth")
    base_cfg.teacher.rec_model_checkpoint_path = ckpt_path
    
    from src.student.models import SASRec
    rec_model = SASRec(num_items=100, hidden_size=32, num_heads=2, num_layers=2, dropout_rate=0.0, max_seq_len=20, padding_item_id=0)
    save_dict = {'state_dict': {f'model.{k}': v for k, v in rec_model.state_dict().items()}}
    torch.save(save_dict, ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})

    try:
        # Create Model (Shared for all tests)
        model = create_teacher_model(
            base_cfg,
            llm_tokenizer=tokenizer,
            num_items=100,
            max_seq_len=20,
            item_id_to_name={i: str(i) for i in range(101)},
            padding_item_id=0,
            candidate_topk=5
        )
        
        # Manually modify embeddings to create a difference
        with torch.no_grad():
            # Modify item 1 significantly
            model.rec_model.item_embeddings.weight[1] += 1.0
            # Modify item 2 slightly
            model.rec_model.item_embeddings.weight[2] += 0.1
            
        current = model.rec_model.item_embeddings.weight
        original = model.student_item_embeddings
            # Common args
        lr = 1e-3
        wd = 0.0
        metrics_k = [10]
        id2name = {i: str(i) for i in range(101)}

        # Test 1: MSE
        trainer_mse = iLoRATrainer(model, 100, lr, wd, metrics_k, id2name, distill_lambda=1.0, distill_loss_type="mse")
        loss_mse = trainer_mse._compute_distill_loss(current, original)
        expected_mse = F.mse_loss(current, original)
        assert torch.isclose(loss_mse, expected_mse), f"MSE mismatch: {loss_mse} vs {expected_mse}"
        print(f"MSE Loss: {loss_mse.item()}")

        # Test 2: L1
        trainer_l1 = iLoRATrainer(model, 100, lr, wd, metrics_k, id2name, distill_lambda=1.0, distill_loss_type="l1")
        loss_l1 = trainer_l1._compute_distill_loss(current, original)
        expected_l1 = F.l1_loss(current, original)
        assert torch.isclose(loss_l1, expected_l1), f"L1 mismatch: {loss_l1} vs {expected_l1}"
        print(f"L1 Loss: {loss_l1.item()}")

        # Test 3: Huber
        trainer_huber = iLoRATrainer(model, 100, lr, wd, metrics_k, id2name, distill_lambda=1.0, distill_loss_type="huber")
        loss_huber = trainer_huber._compute_distill_loss(current, original)
        expected_huber = F.huber_loss(current, original)
        assert torch.isclose(loss_huber, expected_huber), f"Huber mismatch: {loss_huber} vs {expected_huber}"
        print(f"Huber Loss: {loss_huber.item()}")

        # Test 4: Cosine
        trainer_cosine = iLoRATrainer(model, 100, lr, wd, metrics_k, id2name, distill_lambda=1.0, distill_loss_type="cosine")
        loss_cosine = trainer_cosine._compute_distill_loss(current, original)
        cos_sim = F.cosine_similarity(current, original, dim=-1)
        expected_cosine = 1.0 - cos_sim.mean()
        assert torch.isclose(loss_cosine, expected_cosine), f"Cosine mismatch: {loss_cosine} vs {expected_cosine}"
        print(f"Cosine Loss: {loss_cosine.item()}")

        # Test 5: Contrastive
        trainer_contrastive = iLoRATrainer(model, 100, lr, wd, metrics_k, id2name, distill_lambda=1.0, distill_loss_type="contrastive")
        loss_contrastive = trainer_contrastive._compute_distill_loss(current, original)
        
        # Verify it runs and returns a scalar
        assert loss_contrastive.ndim == 0
        assert loss_contrastive > 0
        print(f"Contrastive Loss: {loss_contrastive.item()}")
        
        # Verify gradient flow for one case (e.g. Contrastive)
        loss_contrastive.backward()
        assert model.rec_model.item_embeddings.weight.grad is not None
        assert torch.norm(model.rec_model.item_embeddings.weight.grad) > 0

    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
