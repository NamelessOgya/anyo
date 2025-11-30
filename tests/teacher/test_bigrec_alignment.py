import unittest
import torch
from unittest.mock import MagicMock, patch
from src.data.collators import BigRecCollator
from src.teacher.bigrec_model import BigRecModel

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.pad_token = "<unk>"
        self.eos_token_id = 2
        self.eos_token = "</s>"
        self.padding_side = "left"
    
    def __call__(self, text, return_tensors=None, padding=True, truncation=True, max_length=None, add_special_tokens=True):
        # Mock tokenization
        if isinstance(text, str):
            text = [text]
        
        # If return_tensors is None, return lists (standard AutoTokenizer behavior)
        if return_tensors is None:
            # Return dict of lists
            return {
                "input_ids": [[1] * 10 for _ in text],
                "attention_mask": [[1] * 10 for _ in text]
            }
            
        # If return_tensors="pt", return MagicMock with tensors
        input_ids = torch.ones((len(text), 10), dtype=torch.long) # Dummy
        attention_mask = torch.ones((len(text), 10), dtype=torch.long)
        
        # Return a MagicMock that behaves like the result
        mock_result = MagicMock()
        mock_result.input_ids = input_ids
        mock_result.attention_mask = attention_mask
        mock_result.to.return_value = mock_result # Support .to() chaining
        return mock_result

    def decode(self, token_ids, skip_special_tokens=True):
        return "Dummy Decoded Text"
    
    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ['"Movie A"', '"Movie B"'] # Return quoted strings for testing stripping

class TestBigRecAlignment(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MockTokenizer()

    def test_bigrec_collator_prompt_format(self):
        """
        Verify exact prompt formatting including quotes, intro, suffix, and template spacing.
        """
        item_id_to_name = {1: "Matrix", 2: "Inception", 3: "Interstellar"}
        collator = BigRecCollator(
            tokenizer=self.mock_tokenizer,
            item_id_to_name=item_id_to_name,
            max_source_length=512,
            max_target_length=64,
            train_on_inputs=True
        )
        
        # Wrap tokenizer to spy on calls
        collator.tokenizer = MagicMock(wraps=self.mock_tokenizer)
        # We need to ensure the wrapper has the same attributes
        collator.tokenizer.pad_token_id = self.mock_tokenizer.pad_token_id
        collator.tokenizer.eos_token = self.mock_tokenizer.eos_token
        
        batch = [{
            "seq_ids": [1, 2],
            "next_item_id": 3,
            "history_str": "Unused", # Collator reconstructs from seq_ids
            "candidates_str": "Unused",
            "candidates": [],
            "has_teacher_target": False,
            "teacher_targets": {}
        }]
        
        collator(batch)
        
        # Get the call args
        # collator calls tokenizer multiple times.
        # We want the call for prompt tokenization or full text.
        # With train_on_inputs=True, it tokenizes full text.
        # Check all calls.
        
        found_prompt = False
        found_target = False
        
        expected_intro = 'The user has watched the following movies before: "Matrix", "Inception"\n '
        expected_template_part = "Write a response that appropriately completes the request. \n\n"
        expected_target = '"Interstellar"'
        
        for call in collator.tokenizer.call_args_list:
            args = call[0]
            if len(args) > 0 and isinstance(args[0], str):
                text = args[0]
                # Check for Prompt
                if "The user has watched" in text:
                    found_prompt = True
                    self.assertIn(expected_intro, text, "Intro text or history formatting mismatch")
                    self.assertIn(expected_template_part, text, "Template spacing mismatch")
                    self.assertTrue(text.endswith("### Response:\n"), "Prompt should end with Response tag")
                
                # Check for Target
                if expected_target in text:
                    found_target = True
                    # Target might have EOS appended
                    self.assertTrue(text.startswith(expected_target), "Target text should start with quoted item")
        
        self.assertTrue(found_prompt, "Did not find tokenizer call with prompt text")
        self.assertTrue(found_target, "Did not find tokenizer call with target text")

    def test_bigrec_collator_train_on_inputs(self):
        """
        Verify that labels are NOT masked when train_on_inputs=True.
        """
        item_id_to_name = {1: "A"}
        collator = BigRecCollator(
            tokenizer=self.mock_tokenizer,
            item_id_to_name=item_id_to_name,
            max_source_length=512,
            max_target_length=64,
            train_on_inputs=True
        )
        
        batch = [{"seq_ids": [1], "next_item_id": 1}]
        output = collator(batch)
        
        labels = output["labels"]
        # If train_on_inputs=True, labels should be a copy of input_ids (except padding)
        # They should NOT be -100 everywhere.
        self.assertTrue((labels != -100).any(), "Labels should not be all masked")
        
        # Check that it matches input_ids
        input_ids = output["input_ids"]
        # Ignore padding (which might be masked)
        mask = input_ids != self.mock_tokenizer.pad_token_id
        self.assertTrue(torch.equal(labels[mask], input_ids[mask]), "Labels should match input_ids for non-padding tokens")

    def test_bigrec_model_quote_stripping(self):
        """
        Verify that _evaluate_step strips quotes from generated text.
        """
        # Mock BigRecModel to avoid loading actual model
        with patch("src.teacher.bigrec_model.AutoModelForCausalLM") as mock_model_cls, \
             patch("src.teacher.bigrec_model.AutoTokenizer") as mock_tokenizer_cls, \
             patch("src.teacher.bigrec_model.get_peft_model") as mock_get_peft:
            
            # Mock get_peft_model to just return the model passed to it
            mock_get_peft.side_effect = lambda model, config: model
            
            mock_model = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model
            
            mock_tokenizer = MockTokenizer()
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            model = BigRecModel(model_name_or_path="dummy")
            # Wrap tokenizer to spy
            model.tokenizer = MagicMock(wraps=mock_tokenizer)
            model.tokenizer.pad_token_id = mock_tokenizer.pad_token_id
            model.tokenizer.eos_token_id = mock_tokenizer.eos_token_id
            model.tokenizer.batch_decode = mock_tokenizer.batch_decode # Ensure method is accessible
            
            # Mock generate output (Length 2: 1 input + 1 new)
            model.model.generate.return_value = torch.tensor([[1, 2]])
            
            # Mock compute_item_embeddings to avoid loading file
            model.item_embeddings = torch.randn(10, 32)
            
            batch = {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
                "next_item": torch.tensor([1]),
                "prompt_input_ids": torch.tensor([[1]]), # Needed for generate
                "prompt_attention_mask": torch.tensor([[1]]), # Needed for generate
                "labels": torch.tensor([[1]]) # Needed for loss calculation
            }
            
            # Mock forward pass for loss calculation AND embedding extraction
            # hidden_states should be a list/tuple. We need the last one to be a tensor.
            dummy_hidden = torch.randn(1, 10, 32) # (Batch, Seq, Dim)
            mock_output = MagicMock()
            mock_output.loss = torch.tensor(0.5)
            mock_output.hidden_states = (dummy_hidden,) # Tuple with one element
            model.model.return_value = mock_output
            
            try:
                model._evaluate_step(batch, 0)
            except Exception as e:
                print(f"Caught exception during _evaluate_step: {e}")
                pass
            
            # Inspect calls
            found_stripped = False
            print("\nTokenizer calls:")
            for call in model.tokenizer.call_args_list:
                args = call[0]
                print(f"Args: {args}")
                if isinstance(args[0], list) and "Movie A" in args[0]:
                    # Check if quotes are present
                    if '"Movie A"' not in args[0]:
                        found_stripped = True
                        break
            
            self.assertTrue(found_stripped, "Tokenizer should be called with stripped text (without quotes)")

if __name__ == '__main__':
    unittest.main()
