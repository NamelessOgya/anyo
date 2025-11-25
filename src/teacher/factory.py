from omegaconf import DictConfig
from src.teacher.interfaces import TeacherModel
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
import torch
import torch.nn as nn
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.student.models import SASRec
import logging

logger = logging.getLogger(__name__)

def create_teacher_model(cfg: DictConfig, llm_tokenizer: AutoTokenizer, num_items: int, max_seq_len: int, item_id_to_name: Dict[int, str], padding_item_id: int, candidate_topk: int) -> TeacherModel:
    """
    Hydraの設定に基づいて教師モデルのインスタンスを生成します。
    注: QLoRAのサポートは、カスタムMoeLoraModelとの互換性問題のため一時的に無効化されています。
    """
    torch.set_float32_matmul_precision('high')
    model_type = cfg.teacher.model_type

    if model_type == "ilora":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading LLM for teacher model...")
        
        llm_load_kwargs = {}
        
        if cfg.teacher.get("use_flash_attention", False):
            print("Flash Attention 2 is enabled for LLM.")
            llm_load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            print("Flash Attention is disabled for LLM.")

        if cfg.teacher.get("use_qlora", False):
            # QLoRA is temporarily disabled due to incompatibility with custom MoeLoraModel
            # logger.warning("QLoRA is configured but temporarily disabled.")
            print("QLoRA is configured but temporarily disabled.")

        if cfg.train.get("precision") == "bf16-mixed":
             llm_load_kwargs["torch_dtype"] = torch.bfloat16

        print("DEBUG: Attempting to load LLM from_pretrained...")
        llm = AutoModelForCausalLM.from_pretrained(cfg.teacher.llm_model_name, **llm_load_kwargs)
        logger.info(f"Successfully loaded LLM: {llm.config._name_or_path}")

        # Removed internal tokenizer creation and special token handling.
        # llm.resize_token_embeddings will use the passed llm_tokenizer.
        llm.resize_token_embeddings(len(llm_tokenizer))

        if cfg.teacher.get("rec_model_checkpoint_path"):
            print(f"Loading pre-trained SASRec model from {cfg.teacher.rec_model_checkpoint_path}")
            rec_model = SASRec(
                num_items=num_items,
                hidden_size=cfg.student.hidden_size,
                num_heads=cfg.student.num_heads,
                num_layers=cfg.student.num_layers,
                dropout_rate=cfg.student.dropout_rate,
                max_seq_len=max_seq_len,
                padding_item_id=padding_item_id,
            )
            checkpoint = torch.load(cfg.teacher.rec_model_checkpoint_path, map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('model.'):
                    new_state_dict[k[len('model.'):]] = v
                else:
                    new_state_dict[k] = v
            rec_model.load_state_dict(new_state_dict)
            rec_model.eval()
            rec_model.to(device)
        else:
            raise ValueError("rec_model_checkpoint_path must be provided in the teacher config for iLoRAModel.")
        
        for param in rec_model.parameters():
            param.requires_grad = False
        print("SASRec model parameters frozen.")
        
        projector = MLPProjector(
            input_dim=cfg.student.hidden_size,
            output_dim=llm.config.hidden_size,
            hidden_size=cfg.teacher.hidden_size,
            dropout_rate=cfg.teacher.dropout_rate
        ).to(device, dtype=llm.dtype) # Cast projector to llm.dtype

        model = iLoRAModel(
            llm=llm,
            tokenizer=llm_tokenizer, # Pass the received llm_tokenizer
            num_lora_experts=cfg.teacher.num_lora_experts,
            lora_r=cfg.teacher.lora_r,
            lora_alpha=cfg.teacher.lora_alpha,
            lora_dropout=cfg.teacher.lora_dropout,
            num_items=num_items,
            hidden_size=cfg.teacher.hidden_size,
            dropout_rate=cfg.teacher.dropout_rate,
            rec_model=rec_model,
            projector=projector,
            candidate_topk=candidate_topk,
            item_id_to_name=item_id_to_name,
            padding_item_id=padding_item_id,
            llm_dtype=llm.dtype, # Pass the LLM's dtype
        )

        if cfg.teacher.get("use_torch_compile", False):
            print("Compiling teacher model with torch.compile...")
            model = torch.compile(model)
            
        return model
    else:
        raise ValueError(f"Unknown teacher model type: {model_type}")

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "facebook/opt-125m",
            "num_lora_experts": 3,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "hidden_size": 64,
            "dropout_rate": 0.1,
            "rec_model_checkpoint_path": "path/to/dummy_checkpoint.ckpt" # Added dummy path
        },
        "student": {
            "hidden_size": 64,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.1,
            "max_seq_len": 50,
        },
        "train": {
            "precision": "32"
        }
    })

    num_items_dummy = 5000
    max_seq_len_dummy = 50
    dummy_item_id_to_name = {i: f"Item {i}" for i in range(num_items_dummy + 1)}
    
    # Create a dummy checkpoint file for rec_model
    dummy_rec_model_state = {
        'state_dict': {f'model.layer_{i}': torch.randn(10, 10) for i in range(2)}
    }
    torch.save(dummy_rec_model_state, cfg.teacher.rec_model_checkpoint_path)

    teacher_model = create_teacher_model(
        cfg, 
        num_items_dummy, 
        max_seq_len_dummy, 
        dummy_item_id_to_name,
        padding_item_id=0,
        candidate_topk=10
    )
    print(f"Created teacher model type: {type(teacher_model)}")
    assert isinstance(teacher_model, TeacherModel)
    assert isinstance(teacher_model, torch.nn.Module)

    # Clean up dummy checkpoint
    import os
    os.remove(cfg.teacher.rec_model_checkpoint_path)

    print("\nTeacher model factory test passed!")

