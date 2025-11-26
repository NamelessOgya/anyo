
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import torch
from src.teacher.factory import create_teacher_model

@patch("src.teacher.factory.torch.compile")
@patch("src.teacher.factory.AutoModelForCausalLM")
@patch("src.teacher.factory.AutoTokenizer")
@patch("src.teacher.factory.SASRec")
@patch("torch.load")
def test_create_teacher_model_torch_compile(mock_torch_load, mock_sasrec, mock_tokenizer_cls, mock_automodel_cls, mock_torch_compile):
    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.config.hidden_size = 768
    mock_llm.dtype = torch.float32
    mock_automodel_cls.from_pretrained.return_value = mock_llm
    
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    mock_rec_model = MagicMock()
    mock_sasrec.return_value = mock_rec_model
    
    mock_torch_load.return_value = {'state_dict': {}}

    # Case 1: use_torch_compile = True
    cfg_true = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "dummy/model",
            "num_lora_experts": 2,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "hidden_size": 32,
            "dropout_rate": 0.1,
            "rec_model_checkpoint_path": "dummy_path.ckpt",
            "use_flash_attention": False,
            "use_qlora": False,
            "use_torch_compile": True
        },
        "student": {
            "hidden_size": 32,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.1
        },
        "train": {
            "precision": "bf16-mixed"
        }
    })

    create_teacher_model(
        cfg=cfg_true,
        llm_tokenizer=mock_tokenizer,
        num_items=100,
        max_seq_len=50,
        item_id_to_name={1: "Item 1"},
        padding_item_id=0,
        candidate_topk=10
    )

    assert mock_torch_compile.called, "torch.compile should be called when use_torch_compile=True"

    # Reset mocks
    mock_torch_compile.reset_mock()

    # Case 2: use_torch_compile = False
    cfg_false = OmegaConf.create({
        "teacher": {
            "model_type": "ilora",
            "llm_model_name": "dummy/model",
            "num_lora_experts": 2,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
            "hidden_size": 32,
            "dropout_rate": 0.1,
            "rec_model_checkpoint_path": "dummy_path.ckpt",
            "use_flash_attention": False,
            "use_qlora": False,
            "use_torch_compile": False
        },
        "student": {
            "hidden_size": 32,
            "num_heads": 2,
            "num_layers": 2,
            "dropout_rate": 0.1
        },
        "train": {
            "precision": "bf16-mixed"
        }
    })

    create_teacher_model(
        cfg=cfg_false,
        llm_tokenizer=mock_tokenizer,
        num_items=100,
        max_seq_len=50,
        item_id_to_name={1: "Item 1"},
        padding_item_id=0,
        candidate_topk=10
    )

    assert not mock_torch_compile.called, "torch.compile should NOT be called when use_torch_compile=False"
