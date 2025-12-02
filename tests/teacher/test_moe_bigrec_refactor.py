import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.teacher.moe_bigrec_model import MoEBigRecModel

class TestMoEBigRecRefactor(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/bert_uncased_L-2_H-128_A-2" # Dummy small model
        self.student_path = "dummy_student.ckpt"
        self.item_emb_path = "dummy_item_emb.pt"
        
        # We will mock torch.load, so no need to create real files
        # But we need to ensure os.path.exists returns True for these
        self.patcher_exists = patch("os.path.exists")
        self.mock_exists = self.patcher_exists.start()
        self.mock_exists.side_effect = lambda p: p in [self.student_path, self.item_emb_path] or os.path.exists(p)

    def tearDown(self):
        self.patcher_exists.stop()

    @patch("src.teacher.moe_bigrec_model.get_peft_model")
    @patch("src.teacher.moe_bigrec_model.torch.load")
    @patch("src.teacher.moe_bigrec_model.AutoModelForCausalLM")
    @patch("src.teacher.moe_bigrec_model.AutoTokenizer")
    @patch("src.teacher.moe_bigrec_model.SASRec")
    def test_initialization_requires_sasrec(self, mock_sasrec, mock_tokenizer, mock_lm, mock_load, mock_peft):
        # Setup mocks
        mock_load.return_value = {"state_dict": {}} # Mock checkpoint
        
        # Should raise ValueError if student_model_path is missing
        with self.assertRaises(ValueError):
            MoEBigRecModel(
                model_name_or_path=self.model_name,
                item_embeddings_path=self.item_emb_path,
                student_model_path=None
            )

    @patch("src.teacher.moe_bigrec_model.get_peft_model")
    @patch("src.teacher.moe_bigrec_model.torch.load")
    @patch("src.teacher.moe_bigrec_model.AutoModelForCausalLM")
    @patch("src.teacher.moe_bigrec_model.AutoTokenizer")
    @patch("src.teacher.moe_bigrec_model.SASRec")
    def test_initialization_requires_item_embeddings(self, mock_sasrec, mock_tokenizer, mock_lm, mock_load, mock_peft):
        # Setup mocks
        mock_load.return_value = {"state_dict": {}}
        
        # Should raise ValueError if item_embeddings_path is missing
        with self.assertRaises(ValueError):
            MoEBigRecModel(
                model_name_or_path=self.model_name,
                item_embeddings_path=None,
                student_model_path=self.student_path
            )

    @patch("src.teacher.moe_bigrec_model.get_peft_model")
    @patch("src.teacher.moe_bigrec_model.torch.load")
    @patch("src.teacher.moe_bigrec_model.AutoModelForCausalLM")
    @patch("src.teacher.moe_bigrec_model.AutoTokenizer")
    @patch("src.teacher.moe_bigrec_model.SASRec")
    def test_validation_step_calls_ensemble(self, mock_sasrec_cls, mock_tokenizer, mock_lm, mock_load, mock_peft):
        # Setup mocks
        mock_sasrec_instance = MagicMock()
        mock_sasrec_cls.return_value = mock_sasrec_instance
        # Mock predict to return logits of shape (B, NumItems)
        # Assume batch size 2, NumItems 5 (matching item embeddings)
        mock_sasrec_instance.predict.return_value = torch.randn(2, 5)
        mock_sasrec_instance.return_value = torch.randn(2, 64) # User emb

        # Mock torch.load for item embeddings and checkpoint
        # Item embeddings: {1..5: tensor} -> Max ID 5 -> Tensor (6, 64)
        mock_load.side_effect = [
            {i: torch.randn(64) for i in range(1, 6)}, # item embeddings (5 items)
            {"state_dict": {}} # checkpoint
        ]

        model = MoEBigRecModel(
            model_name_or_path=self.model_name,
            item_embeddings_path=self.item_emb_path,
            student_model_path=self.student_path,
            metrics_k=5
        )
        
        # Mock model forward for LLM logits
        # LLM user emb: (B, H) -> (2, 64)
        # Item emb: (6, 64)
        # LLM logits full: (2, 6). Remove padding -> (2, 5).
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.randn(2, 10, 64)] # Last hidden state (B, Seq, H)
        model.model.model.return_value = mock_output
        
        # Mock self.log
        model.log = MagicMock()
        
        # Create dummy batch
        batch = {
            "prompt_input_ids": torch.randint(0, 100, (2, 10)),
            "prompt_attention_mask": torch.ones(2, 10),
            "sasrec_input_ids": torch.randint(0, 100, (2, 10)),
            "next_item": torch.tensor([1, 2]) # Targets
        }
        
        # Run validation_step
        model.validation_step(batch, 0)
        
        # Check if log was called with expected keys
        # We expect val_hr@5, val_ndcg@5, val_alpha
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        self.assertIn("val_hr@5", logged_keys)
        self.assertIn("val_ndcg@5", logged_keys)
        self.assertIn("val_alpha", logged_keys)

if __name__ == "__main__":
    unittest.main()
