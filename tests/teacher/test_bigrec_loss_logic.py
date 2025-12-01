import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from unittest.mock import MagicMock, patch
from src.teacher.moe_bigrec_model import MoEBigRecModel

class TestBigRecLossLogic(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
        
        with patch('src.teacher.moe_bigrec_model.AutoTokenizer.from_pretrained', return_value=self.mock_tokenizer), \
             patch('src.teacher.moe_bigrec_model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.teacher.moe_bigrec_model.get_peft_model'), \
             patch('os.path.exists', return_value=True), \
             patch('torch.load', return_value=torch.zeros(3)):
            
            self.model = MoEBigRecModel(
                model_name_or_path="dummy",
                student_model_path="dummy_sasrec", # Trigger ensemble logic
                popularity_path="dummy_pop", # Trigger popularity logic
                popularity_lambda=10.0 # High lambda to make effect obvious
            )
            
        # Manually setup components that are usually loaded
        # self.model.device is read-only property in PL
        self.model.item_embeddings = torch.tensor([
            [0.0, 0.0], # Padding
            [1.0, 0.0], # Item 1
            [0.0, 1.0]  # Item 2
        ]) # 2 Items, 2 Dim
        
        # Mock SASRec
        self.model.sasrec = MagicMock()
        
        # Mock Gate
        self.model.gate = nn.Linear(2, 1)
        # Set gate weights to produce alpha=0.5
        # Sigmoid(0) = 0.5. So we want output 0.
        nn.init.zeros_(self.model.gate.weight)
        nn.init.zeros_(self.model.gate.bias)
        
        # Mock Popularity Scores
        # Item 1: count 0 -> log(1) = 0
        # Item 2: count e-1 -> log(e) = 1
        import math
        self.model.popularity_scores = torch.tensor([
            0.0, # Padding
            0.0, # Item 1
            math.e - 1 # Item 2
        ])
        
    def test_loss_calculation(self):
        # Setup Batch
        batch = {
            "prompt_input_ids": torch.zeros((1, 5), dtype=torch.long),
            "prompt_attention_mask": torch.ones((1, 5), dtype=torch.long),
            "sasrec_input_ids": torch.zeros((1, 5), dtype=torch.long),
            "next_item": torch.tensor([1], dtype=torch.long), # Target is Item 1 (Index 0 in logits)
            "input_ids": torch.zeros((1, 5), dtype=torch.long), # Dummy
            "attention_mask": torch.ones((1, 5), dtype=torch.long), # Dummy
            "labels": torch.zeros((1, 5), dtype=torch.long) # Dummy
        }
        
        # Mock LLM Output
        # We want LLM User Emb to be [10.0, 0.0]
        # So LLM Logits (raw) = [10.0, 0.0] @ [[1,0], [0,1]].T = [10.0, 0.0]
        mock_outputs = MagicMock()
        # hidden_states: tuple of (B, Seq, H). We need last layer.
        # Shape: (1, 5, 2)
        last_hidden = torch.zeros(1, 5, 2)
        last_hidden[:, -1, :] = torch.tensor([10.0, 0.0])
        mock_outputs.hidden_states = (last_hidden,)
        self.model.model.model.return_value = mock_outputs
        
        # Mock SASRec Output
        # SASRec Logits: [0.0, 10.0]
        self.model.sasrec.predict.return_value = torch.tensor([[0.0, 10.0]])
        # SASRec User Emb: [0.0, 0.0] (doesn't matter for gate since weights are 0)
        self.model.sasrec.return_value = torch.zeros(1, 2)
        
        # --- Expected Calculation ---
        # 1. Popularity Adjustment (LLM)
        # Pop Scores (1..N): [0.0, e-1]
        # Log(Pop+1): [0.0, 1.0]
        # Lambda: 10.0
        # Adjustment: [0.0, 10.0]
        # LLM Logits (Raw): [10.0, 0.0]
        # LLM Logits (Adj): [10.0, 0.0] + [0.0, 10.0] = [10.0, 10.0]
        
        # 2. Min-Max Normalization
        # LLM (Adj): [10.0, 10.0]
        # Min: 10.0, Max: 10.0
        # Norm: (10 - 10) / (10 - 10 + 1e-8) = 0.0
        # LLM Norm: [0.0, 0.0]
        
        # SASRec: [0.0, 10.0]
        # Min: 0.0, Max: 10.0
        # Norm: (x - 0) / (10 - 0 + 1e-8) ~ x / 10
        # SASRec Norm: [0.0, 1.0]
        
        # 3. Loss Calculation
        # Target: Item 1 (Index 0)
        # LLM Loss: CrossEntropy([0.0, 0.0], 0)
        # Softmax([0, 0]) = [0.5, 0.5]
        # Loss = -log(0.5) = 0.693147
        
        # SASRec Loss: CrossEntropy([0.0, 1.0], 0)
        # Softmax([0, 1]) = [1/(1+e), e/(1+e)] = [0.26894, 0.73105]
        # Loss = -log(0.26894) = 1.31326
        
        # 4. Mixture
        # Alpha = Sigmoid(0) = 0.5
        # Total Loss = 0.5 * 1.31326 + 0.5 * 0.693147
        #            = 0.65663 + 0.34657
        #            = 1.0032
        
        expected_loss = 1.0032
        
        # Run Training Step
        loss = self.model.training_step(batch, 0)
        
        print(f"Calculated Loss: {loss.item()}")
        print(f"Expected Loss: {expected_loss}")
        
        self.assertAlmostEqual(loss.item(), expected_loss, places=3)

if __name__ == '__main__':
    unittest.main()
