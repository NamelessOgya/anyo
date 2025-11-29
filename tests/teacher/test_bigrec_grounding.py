import unittest
from unittest.mock import MagicMock, patch
import torch
from src.teacher.bigrec_model import BigRecModel

class TestBigRecGrounding(unittest.TestCase):
    @patch('src.teacher.bigrec_model.AutoTokenizer')
    @patch('src.teacher.bigrec_model.AutoModelForCausalLM')
    @patch('src.teacher.bigrec_model.get_peft_model')
    @patch('torch.load')
    @patch('os.path.exists')
    def test_grounding_validation(self, mock_exists, mock_load, mock_get_peft, mock_automodel, mock_autotokenizer):
        # Setup Mocks
        mock_exists.return_value = True
        
        # Mock Item Embeddings (2 items, dim=4)
        # Item 0: [1, 0, 0, 0]
        # Item 1: [0, 1, 0, 0]
        item_embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        mock_load.return_value = item_embeddings
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model
        mock_get_peft.return_value = mock_model
        
        # Mock generate output
        # Batch size 2, Top-1 generation
        # Batch 0: Generates text that embeds to [1, 0, 0, 0] (Item 0)
        # Batch 1: Generates text that embeds to [0, 0, 1, 0] (Far from both)
        
        # Mock generate return value (Batch, SeqLen)
        # Just dummy IDs
        generated_ids = torch.zeros((2, 10), dtype=torch.long)
        mock_model.generate.return_value = generated_ids
        
        # Mock batch_decode
        mock_tokenizer.batch_decode.return_value = ["Text A", "Text B"]
        
        # Mock tokenizer call for embedding generated text
        # The code calls tokenizer(...).to(device)
        # So we need to mock the return value of .to()
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.zeros((2, 5), dtype=torch.long)
        mock_encoding.attention_mask = torch.ones((2, 5), dtype=torch.long)
        
        mock_tokenizer.return_value.to.return_value = mock_encoding
        
        # Mock base model forward (disable_adapter context)
        # We need to mock the context manager
        mock_model.disable_adapter.return_value.__enter__.return_value = None
        
        # Mock model output for text embedding
        # Batch 0: [1, 0, 0, 0] -> Close to Item 0
        # Batch 1: [0, 0, 1, 0] -> Far from Item 0 and 1
        # We need seq_len=5 to match attention_mask
        last_hidden = torch.zeros((2, 5, 4))
        # Set the last token (index 4) embedding
        last_hidden[0, 4, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        last_hidden[1, 4, :] = torch.tensor([0.0, 0.9, 0.0, 0.0]) # Updated for second run logic
        
        mock_output = MagicMock()
        mock_output.hidden_states = [None, last_hidden] # last element is used
        mock_model.return_value = mock_output
        
        # Instantiate Model
        model = BigRecModel(
            model_name_or_path="dummy",
            item_embeddings_path="dummy_path.pt",
            metrics_k=1
        )
        
        # Mock log
        model.log = MagicMock()
        
        # Create Batch
        batch = {
            "input_ids": torch.zeros((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "labels": torch.zeros((2, 5), dtype=torch.long),
            "next_item": torch.tensor([0, 1]) # Target: Item 0, Item 1
        }
        
        # Run validation step
        model.validation_step(batch, 0)
        
        # Verify Metrics
        # Batch 0: Pred [1,0,0,0]. Target Item 0 [1,0,0,0]. Dist 0. Hit.
        # Batch 1: Pred [0,0,1,0]. Target Item 1 [0,1,0,0]. Dist sqrt(2).
        # Item 0 dist to Pred: sqrt(1+1)=1.41
        # Item 1 dist to Pred: sqrt(1)=1.0
        # If K=1, Batch 1 picks Item 1 (closest). So Hit?
        # Wait, Pred [0,0,1,0].
        # Dist to Item 0 [1,0,0,0]: sqrt(1+0+1) = 1.414
        # Dist to Item 1 [0,1,0,0]: sqrt(0+1+1) = 1.414
        # Tie.
        # Let's make Batch 1 Pred closer to Item 1.
        # Pred [0, 0.9, 0, 0].
        
        # Update mock return
        last_hidden = torch.zeros((2, 5, 4))
        last_hidden[0, 4, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        last_hidden[1, 4, :] = torch.tensor([0.0, 0.9, 0.0, 0.0])
        
        mock_output.hidden_states = [None, last_hidden]
        mock_model.return_value = mock_output
        
        # Run again
        model.validation_step(batch, 0)
        
        # Batch 0: Hit.
        # Batch 1: Pred [0, 0.9, 0, 0]. Target Item 1 [0, 1, 0, 0]. Dist 0.1.
        # Item 0 dist: sqrt(1 + 0.81) > 0.1.
        # So Item 1 is closest. Hit.
        
        # HR should be 1.0
        calls = model.log.call_args_list
        hr_call = next((c for c in calls if c[0][0] == "val_hr@1"), None)
        self.assertIsNotNone(hr_call)
        # self.assertAlmostEqual(hr_call[0][1], 1.0) # Might be called twice because we ran twice?
        # Actually we ran twice but mock_model return was updated for the second run? 
        # No, we defined `last_hidden` *after* the first run in the comment but in code it's defined once.
        # Ah, I see. I should just set the correct value first.
        
        self.assertAlmostEqual(hr_call[0][1], 1.0)

    @patch('src.teacher.bigrec_model.AutoTokenizer')
    @patch('src.teacher.bigrec_model.AutoModelForCausalLM')
    @patch('src.teacher.bigrec_model.get_peft_model')
    @patch('torch.load')
    @patch('os.path.exists')
    def test_grounding_test_step(self, mock_exists, mock_load, mock_get_peft, mock_automodel, mock_autotokenizer):
        # Setup Mocks (Same as validation)
        mock_exists.return_value = True
        item_embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        mock_load.return_value = item_embeddings
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model
        mock_get_peft.return_value = mock_model
        
        generated_ids = torch.zeros((2, 10), dtype=torch.long)
        mock_model.generate.return_value = generated_ids
        mock_tokenizer.batch_decode.return_value = ["Text A", "Text B"]
        
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.zeros((2, 5), dtype=torch.long)
        mock_encoding.attention_mask = torch.ones((2, 5), dtype=torch.long)
        mock_tokenizer.return_value.to.return_value = mock_encoding
        
        mock_model.disable_adapter.return_value.__enter__.return_value = None
        
        last_hidden = torch.zeros((2, 5, 4))
        last_hidden[0, 4, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        last_hidden[1, 4, :] = torch.tensor([0.0, 1.0, 0.0, 0.0]) # Hit for Item 1
        
        mock_output = MagicMock()
        mock_output.hidden_states = [None, last_hidden]
        mock_model.return_value = mock_output
        
        model = BigRecModel(model_name_or_path="dummy", item_embeddings_path="dummy_path.pt", metrics_k=1)
        model.log = MagicMock()
        
        batch = {
            "input_ids": torch.zeros((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
            "labels": torch.zeros((2, 5), dtype=torch.long),
            "next_item": torch.tensor([0, 1])
        }
        
        # Run test_step
        model.test_step(batch, 0)
        
        # Verify Metrics
        calls = model.log.call_args_list
        hr_call = next((c for c in calls if c[0][0] == "test_hr@1"), None)
        self.assertIsNotNone(hr_call)
        self.assertAlmostEqual(hr_call[0][1], 1.0)

if __name__ == '__main__':
    unittest.main()
