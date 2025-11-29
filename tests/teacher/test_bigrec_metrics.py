import unittest
from unittest.mock import MagicMock, patch
import torch
from src.teacher.bigrec_model import BigRecModel

class TestBigRecMetrics(unittest.TestCase):
    @patch('src.teacher.bigrec_model.AutoTokenizer')
    @patch('src.teacher.bigrec_model.AutoModelForCausalLM')
    @patch('src.teacher.bigrec_model.get_peft_model')
    def test_validation_metrics(self, mock_get_peft, mock_automodel, mock_autotokenizer):
        # Setup Mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_autotokenizer.from_pretrained.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model
        mock_get_peft.return_value = mock_model
        
        # Mock generate output
        # Batch size 2, K=2
        # Target: "Item A", "Item B"
        # Generated: [["Item A", "Item C"], ["Item D", "Item E"]] -> Batch 0 Hit, Batch 1 Miss
        
        # Mock decode
        def mock_batch_decode(ids, **kwargs):
            # ids is (K, SeqLen) tensor
            results = []
            for seq in ids:
                # seq is (SeqLen,)
                if (seq == 10).any(): results.append("Item A")
                elif (seq == 11).any(): results.append("Item B")
                elif (seq == 12).any(): results.append("Item C")
                elif (seq == 13).any(): results.append("Item D")
                else: results.append("Unknown")
            return results
            
        mock_tokenizer.batch_decode.side_effect = mock_batch_decode

        # Instantiate Model
        item_id_to_name = {1: "Item A", 2: "Item B"}
        model = BigRecModel(
            model_name_or_path="dummy",
            item_id_to_name=item_id_to_name,
            metrics_k=2
        )
        
        # Mock generate return value
        # Shape: (Batch*K, SeqLen)
        # Batch 0: [Item A tokens], [Item C tokens]
        # Batch 1: [Item D tokens], [Item E tokens]
        # We assume input_len=2, new_tokens=1
        # Generated IDs should include input_ids? Yes, usually.
        # But our logic slices `new_tokens = generated_ids[i, :, input_len:]`
        # So we need to provide tensor with length > input_len
        
        batch_size = 2
        k = 2
        input_len = 5
        seq_len = input_len + 1
        
        # Mock generate to return a tensor
        # We need to return (Batch*K, SeqLen)
        # Content doesn't matter much as we mocked batch_decode
        # But batch_decode is called on SLICED tensor.
        # So newly generated tokens are what matters.
        # Let's say new token 10 -> "Item A"
        
        # Batch 0, Beam 0: New token 10 (Item A) -> Hit
        # Batch 0, Beam 1: New token 12 (Item C)
        # Batch 1, Beam 0: New token 13 (Item D) -> Miss
        # Batch 1, Beam 1: New token 13 (Item D)
        
        generated_tensor = torch.zeros((batch_size * k, seq_len), dtype=torch.long)
        # Set new tokens at index input_len
        generated_tensor[0, input_len] = 10 # Item A
        generated_tensor[1, input_len] = 12 # Item C
        generated_tensor[2, input_len] = 13 # Item D
        generated_tensor[3, input_len] = 13 # Item D
        
        model.model.generate.return_value = generated_tensor
        
        # Mock log
        model.log = MagicMock()
        
        # Create Batch
        batch = {
            "input_ids": torch.zeros((batch_size, input_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, input_len), dtype=torch.long),
            "labels": torch.zeros((batch_size, input_len), dtype=torch.long),
            "next_item": torch.tensor([1, 2]) # Target IDs: 1->Item A, 2->Item B
        }
        
        # Run validation step
        model.validation_step(batch, 0)
        
        # Verify Metrics
        # Batch 0: Target "Item A". Preds ["Item A", "Item C"]. Hit. Rank 0. NDCG = 1/log2(2) = 1.0
        # Batch 1: Target "Item B". Preds ["Item D", "Item D"]. Miss. NDCG = 0.0
        # Avg HR = 0.5
        # Avg NDCG = 0.5
        
        # Check log calls
        # model.log("val_hr@2", 0.5, ...)
        # model.log("val_ndcg@2", 0.5, ...)
        
        calls = model.log.call_args_list
        # Find calls for metrics
        hr_call = next((c for c in calls if c[0][0] == "val_hr@2"), None)
        ndcg_call = next((c for c in calls if c[0][0] == "val_ndcg@2"), None)
        
        self.assertIsNotNone(hr_call)
        self.assertAlmostEqual(hr_call[0][1], 0.5)
        
        self.assertIsNotNone(ndcg_call)
        self.assertAlmostEqual(ndcg_call[0][1], 0.5)

if __name__ == '__main__':
    unittest.main()
