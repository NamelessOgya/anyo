
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import torch
from src.teacher.factory import create_teacher_model

@patch("src.teacher.factory.AutoModelForCausalLM")
@patch("src.teacher.factory.AutoTokenizer")
@patch("src.teacher.factory.SASRec")
@patch("torch.load")
def test_create_teacher_model_dtype_args(mock_torch_load, mock_sasrec, mock_tokenizer_cls, mock_automodel_cls):
    # Setup mocks
    mock_llm = MagicMock()
    mock_llm.config.hidden_size = 768
    mock_llm.dtype = torch.float32
    mock_automodel_cls.from_pretrained.return_value = mock_llm
    
    mock_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    class Dummy:
        pass
    mock_rec_model = Dummy()
    mock_rec_model.hidden_size = 32
    mock_rec_model.num_items = 100
    mock_rec_model.item_embeddings = torch.nn.Embedding(10, 32)
    mock_rec_model.parameters = MagicMock(return_value=[])
    mock_rec_model.load_state_dict = MagicMock()
    mock_rec_model.eval = MagicMock()
    mock_rec_model.to = MagicMock(return_value=mock_rec_model)
    
    mock_sasrec.return_value = mock_rec_model
    
    mock_torch_load.return_value = {'state_dict': {}}

    # Create config with bf16-mixed precision
    cfg = OmegaConf.create({
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
            "use_qlora": False
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

    # Call factory
    create_teacher_model(
        cfg=cfg,
        llm_tokenizer=mock_tokenizer,
        num_items=100,
        max_seq_len=50,
        item_id_to_name={1: "Item 1"},
        padding_item_id=0,
        candidate_topk=10
    )

    # Verify arguments passed to from_pretrained
    args, kwargs = mock_automodel_cls.from_pretrained.call_args
    
    assert args[0] == "dummy/model"
    assert "torch_dtype" in kwargs, "torch_dtype should be passed to from_pretrained"
    assert kwargs["torch_dtype"] == torch.bfloat16
    assert "dtype" not in kwargs, "dtype should NOT be passed to from_pretrained (it causes TypeError)"

