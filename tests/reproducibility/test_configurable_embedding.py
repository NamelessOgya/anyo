import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from src.teacher.ilora_model import iLoRAModel
from src.teacher.trainer_ilora import iLoRATrainer
from src.teacher.factory import create_teacher_model
from transformers import AutoTokenizer
import os

def test_configurable_embedding_head():
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
            "rec_model_checkpoint_path": "dummy_ckpt_conf.pth",
            "use_flash_attention": False,
            "use_qlora": False,
            "use_gradient_checkpointing": False,
            "use_torch_compile": False,
            "distill_lambda": 0.1
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
    ckpt_path = os.path.abspath("dummy_ckpt_conf.pth")
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
        # Test Case A: use_item_embeddings_head = True (Default)
        print("Testing use_item_embeddings_head = True")
        cfg_true = base_cfg.copy()
        cfg_true.teacher.use_item_embeddings_head = True
        
        model_true = create_teacher_model(
            cfg_true,
            llm_tokenizer=tokenizer,
            num_items=100,
            max_seq_len=20,
            item_id_to_name={i: str(i) for i in range(101)},
            padding_item_id=0,
            candidate_topk=5
        )
        
        assert model_true.use_item_embeddings_head == True
        assert model_true.item_prediction_head is None
        assert hasattr(model_true, "student_item_embeddings")
        assert model_true.rec_model.item_embeddings.weight.requires_grad == True
        
        # Verify Loss Calculation for True Case
        trainer_true = iLoRATrainer(
            ilora_model=model_true,
            num_items=100,
            learning_rate=1e-3,
            weight_decay=0.0,
            metrics_k=[10],
            item_id_to_name={i: str(i) for i in range(101)},
            distill_lambda=1.0 # High lambda to make effect obvious
        )
        
        # Mock log method to capture logs
        logs_true = {}
        trainer_true.log = lambda name, value, **kwargs: logs_true.update({name: value})
        
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
        
        # Simulate divergence
        with torch.no_grad():
            model_true.rec_model.item_embeddings.weight[1] += 1.0
            
        loss_true = trainer_true.training_step(batch, 0)
        
        assert "train_reg_loss" in logs_true, "train_reg_loss should be logged when enabled"
        assert logs_true["train_reg_loss"] > 0, "train_reg_loss should be positive"
        
        
        # Test Case B: use_item_embeddings_head = False
        print("Testing use_item_embeddings_head = False")
        cfg_false = base_cfg.copy()
        cfg_false.teacher.use_item_embeddings_head = False
        
        model_false = create_teacher_model(
            cfg_false,
            llm_tokenizer=tokenizer,
            num_items=100,
            max_seq_len=20,
            item_id_to_name={i: str(i) for i in range(101)},
            padding_item_id=0,
            candidate_topk=5
        )
        
        assert model_false.use_item_embeddings_head == False
        assert isinstance(model_false.item_prediction_head, nn.Linear)
        assert not hasattr(model_false, "student_item_embeddings")
        assert model_false.rec_model.item_embeddings.weight.requires_grad == False

        # Verify Loss Calculation for False Case
        trainer_false = iLoRATrainer(
            ilora_model=model_false,
            num_items=100,
            learning_rate=1e-3,
            weight_decay=0.0,
            metrics_k=[10],
            item_id_to_name={i: str(i) for i in range(101)},
            distill_lambda=1.0 
        )
        
        # Mock log method
        logs_false = {}
        trainer_false.log = lambda name, value, **kwargs: logs_false.update({name: value})
        
        loss_false = trainer_false.training_step(batch, 0)
        
        assert "train_reg_loss" not in logs_false, "train_reg_loss should NOT be logged when disabled"

    finally:
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
