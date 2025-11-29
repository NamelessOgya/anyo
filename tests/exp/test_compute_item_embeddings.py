import unittest
from unittest.mock import MagicMock, patch
import torch
import pandas as pd
from src.exp.compute_item_embeddings import main
from omegaconf import DictConfig

class TestComputeItemEmbeddings(unittest.TestCase):
    @patch('src.exp.compute_item_embeddings.AutoTokenizer')
    @patch('src.exp.compute_item_embeddings.AutoModelForCausalLM')
    @patch('src.exp.compute_item_embeddings.pd.read_csv')
    @patch('src.exp.compute_item_embeddings.torch.save')
    @patch('src.exp.compute_item_embeddings.Path')
    def test_logic(self, mock_path, mock_save, mock_read_csv, mock_automodel, mock_autotokenizer):
        # Setup Mocks
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.__truediv__.return_value = mock_path_obj # handle / operator
        mock_path_obj.exists.return_value = True
        
        # Mock Data
        mock_df = pd.DataFrame({
            'item_id': [0, 1],
            'title': ['Movie A', 'Movie B']
        })
        mock_read_csv.return_value = mock_df
        
        # Mock Tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer
        
        # Mock Model
        mock_model = MagicMock()
        mock_model.config.hidden_size = 4
        mock_automodel.from_pretrained.return_value = mock_model
        
        # Mock Model Output
        # Batch size 2
        # Output shape: (Batch, Seq, Dim)
        last_hidden = torch.randn(2, 3, 4)
        mock_output = MagicMock()
        mock_output.hidden_states = [None, last_hidden]
        mock_model.return_value = mock_output
        
        # Mock tokenizer call
        mock_tokenizer.return_value = MagicMock(
            input_ids=torch.zeros((2, 3)),
            attention_mask=torch.ones((2, 3))
        )
        
        # Create Config
        cfg = DictConfig({
            "dataset": {"data_dir": "dummy_dir"},
            "teacher": {
                "llm_model_name": "dummy_model",
                "max_target_length": 10
            }
        })
        
        # Run Logic
        from src.exp.compute_item_embeddings import compute_embeddings
        compute_embeddings(cfg)
        
        # Verify Save
        mock_save.assert_called_once()
        # Check saved tensor shape
        # max_item_id is 1. Size should be (2, 4).
        saved_tensor = mock_save.call_args[0][0]
        self.assertEqual(saved_tensor.shape, (2, 4))

if __name__ == '__main__':
    unittest.main()
