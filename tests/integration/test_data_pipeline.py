
import pytest
import pandas as pd
import torch
from unittest.mock import MagicMock
from src.student.datamodule import SASRecDataset
from src.data.collators import BigRecCollator

def test_dataset_collator_compatibility():
    """
    Integration test to verify that SASRecDataset outputs are compatible with BigRecCollator.
    This prevents KeyErrors if keys in dataset or collator change.
    """
    # 1. Setup Dummy Data
    df = pd.DataFrame({
        "user_id": [1, 2],
        "seq": [[1, 2], [3, 4, 5]],
        "next_item": [3, 6],
        "candidates": [[1, 2, 3], [4, 5, 6]] # Not used by BigRecCollator but needed for Dataset init
    })
    
    item_id_to_name = {1: "Item1", 2: "Item2", 3: "Item3", 4: "Item4", 5: "Item5", 6: "Item6"}
    id_to_history_part = {k: f"{v} [HistoryEmb]" for k, v in item_id_to_name.items()}
    id_to_candidate_part = {k: f"{v} [CansEmb]" for k, v in item_id_to_name.items()}
    
    # 2. Instantiate Dataset
    dataset = SASRecDataset(
        df=df,
        max_seq_len=10,
        num_items=6,
        item_id_to_name=item_id_to_name,
        num_candidates=5,
        padding_item_id=0,
        id_to_history_part=id_to_history_part,
        id_to_candidate_part=id_to_candidate_part
    )
    
    # 3. Instantiate Collator
    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    tokenizer.return_value = MagicMock(
        input_ids=torch.ones((1, 10), dtype=torch.long),
        attention_mask=torch.ones((1, 10), dtype=torch.long)
    )
    
    collator = BigRecCollator(
        tokenizer=tokenizer,
        item_id_to_name=item_id_to_name,
        max_source_length=20,
        max_target_length=10,
        use_cot=False
    )
    
    # 4. Fetch Batch and Collate
    batch = [dataset[0], dataset[1]]
    
    try:
        output = collator(batch)
    except KeyError as e:
        pytest.fail(f"Collator failed with KeyError: {e}. Mismatch between Dataset output and Collator expectation.")
    except Exception as e:
        pytest.fail(f"Collator failed with unexpected error: {e}")
        
    # 5. Verify Output Keys
    assert "input_ids" in output
    assert "attention_mask" in output
    assert "labels" in output
    assert "prompt_input_ids" in output
