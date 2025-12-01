import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from src.teacher.bigrec_model import BigRecModel

class TestBigRecEnsemble(unittest.TestCase):
    def setUp(self):
        self.model_name = "gpt2" # Dummy
        self.student_path = "dummy_sasrec.ckpt"
        self.item_emb_path = "dummy_item_emb.pt"
        
    @patch("src.teacher.bigrec_model.AutoModelForCausalLM")
    @patch("src.teacher.bigrec_model.AutoTokenizer")
    @patch("src.teacher.bigrec_model.get_peft_model")
    @patch("src.teacher.bigrec_model.SASRec")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_initialization(self, mock_exists, mock_load, mock_sasrec_cls, mock_get_peft, mock_tokenizer, mock_automodel):
        # Setup mocks
        mock_exists.return_value = True
        
        # Mock Checkpoint
        mock_sasrec_instance = MagicMock()
        mock_sasrec_instance.parameters.return_value = [MagicMock(requires_grad=True)]
        mock_sasrec_cls.return_value = mock_sasrec_instance
        
        mock_checkpoint = {"state_dict": {"model.layer1.weight": torch.randn(10, 10)}}
        mock_load.side_effect = lambda path, map_location=None: mock_checkpoint if path == self.student_path else torch.randn(10, 64) # Item embeddings
        
        # Initialize
        model = BigRecModel(
            model_name_or_path=self.model_name,
            student_model_path=self.student_path,
            item_embeddings_path=self.item_emb_path
        )
        
        # Verify SASRec loaded
        self.assertIsNotNone(model.sasrec)
        mock_sasrec_cls.assert_called_once()
        # Verify frozen
        for param in model.sasrec.parameters():
            self.assertFalse(param.requires_grad)
        # Verify gate
        self.assertIsNotNone(model.gate)
        self.assertIsInstance(model.gate, nn.Linear)
        
    @patch("src.teacher.bigrec_model.AutoModelForCausalLM")
    @patch("src.teacher.bigrec_model.AutoTokenizer")
    @patch("src.teacher.bigrec_model.get_peft_model")
    @patch("src.teacher.bigrec_model.SASRec")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_training_step(self, mock_exists, mock_load, mock_sasrec_cls, mock_get_peft, mock_tokenizer, mock_automodel):
        mock_exists.return_value = True
        
        # Mock SASRec
        mock_sasrec_instance = MagicMock()
        mock_sasrec_instance.parameters.return_value = [MagicMock(requires_grad=True)]
        # Mock predict: (B, NumItems) = (2, 9) (10 embeddings - 1 padding)
        mock_sasrec_instance.predict.return_value = torch.randn(2, 9)
        # Mock forward (user embedding): (B, Hidden) = (2, 64)
        mock_sasrec_instance.return_value = torch.randn(2, 64)
        mock_sasrec_cls.return_value = mock_sasrec_instance
        
        # Mock Item Embeddings (10 items, 32 dim)
        item_embs = torch.randn(10, 32)
        mock_load.side_effect = lambda path, map_location=None: {"state_dict": {}} if path == self.student_path else item_embs
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.model.return_value.hidden_states = [torch.randn(2, 5, 32)] # (B, Seq, H)
        mock_get_peft.return_value = mock_llm
        
        model = BigRecModel(
            model_name_or_path=self.model_name,
            student_model_path=self.student_path,
            item_embeddings_path=self.item_emb_path
        )
        model.item_embeddings = item_embs # Ensure it's set
        
        # Mock Batch
        batch = {
            "prompt_input_ids": torch.randint(0, 100, (2, 5)),
            "prompt_attention_mask": torch.ones(2, 5),
            "sasrec_input_ids": torch.randint(0, 10, (2, 5)),
            "next_item": torch.tensor([1, 2]) # 1-based IDs
        }
        
        # Run training_step
        loss = model.training_step(batch, 0)
        
        # Verify loss is scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        
        # Verify SASRec called
        mock_sasrec_instance.predict.assert_called_once()
        
        # Verify LLM called (model.model)
        mock_llm.model.assert_called_once()

    @patch("src.teacher.bigrec_model.AutoModelForCausalLM")
    @patch("src.teacher.bigrec_model.AutoTokenizer")
    @patch("src.teacher.bigrec_model.get_peft_model")
    @patch("src.teacher.bigrec_model.SASRec")
    @patch("torch.load")
    @patch("os.path.exists")
    def test_validation_step(self, mock_exists, mock_load, mock_sasrec_cls, mock_get_peft, mock_tokenizer, mock_automodel):
        mock_exists.return_value = True
        
        # Mock SASRec
        mock_sasrec_instance = MagicMock()
        mock_sasrec_instance.parameters.return_value = [MagicMock(requires_grad=True)]
        mock_sasrec_instance.predict.return_value = torch.randn(2, 9) # Match item count (9)
        mock_sasrec_instance.return_value = torch.randn(2, 64) # User embeddings
        mock_sasrec_cls.return_value = mock_sasrec_instance
        
        # Mock Item Embeddings
        item_embs = torch.randn(10, 32)
        mock_load.side_effect = lambda path, map_location=None: {"state_dict": {}} if path == self.student_path else item_embs
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.model.return_value.hidden_states = [torch.randn(2, 5, 32)]
        # Mock generate
        mock_llm.generate.return_value = torch.randint(0, 100, (2, 10)) # Input(5) + Gen(5)
        mock_get_peft.return_value = mock_llm
        
        # Mock Tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.eos_token_id = 1
        mock_tokenizer_instance.batch_decode.return_value = ["Generated Text 1", "Generated Text 2"]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        model = BigRecModel(
            model_name_or_path=self.model_name,
            student_model_path=self.student_path,
            item_embeddings_path=self.item_emb_path
        )
        model.item_embeddings = item_embs
        model.log = MagicMock()
        model.metrics_k = 5 # Set K < NumItems (9)
        model.item_id_to_name = {i: f"Item_{i}" for i in range(1, 10)} # Mock item map
        
        # Mock SASRec forward for Gating
        mock_sasrec_instance.return_value = torch.randn(2, 64) # User embeddings
        
        batch = {
            "prompt_input_ids": torch.randint(0, 100, (2, 5)),
            "prompt_attention_mask": torch.ones(2, 5),
            "sasrec_input_ids": torch.randint(0, 10, (2, 5)),
            "next_item": torch.tensor([1, 2])
        }
        
        # Run validation_step with batch_idx=0 to trigger debug logging
        model.validation_step(batch, 0)
        
        # Verify logging
        calls = [c[0][0] for c in model.log.call_args_list]
        self.assertIn("val_hr@5", calls)
        self.assertIn("val_ndcg@5", calls)
        self.assertIn("val_alpha", calls)
        
        # Verify generate called
        mock_llm.generate.assert_called_once()
        mock_tokenizer_instance.batch_decode.assert_called_once()

if __name__ == "__main__":
    unittest.main()
