import pytest
import torch
import pandas as pd
import numpy as np
from src.student.datamodule import SASRecDataset

class TestActiveLearning:
    @pytest.fixture
    def mock_data(self):
        # Create dummy dataframe
        df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'seq': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
            'next_item': [3, 5, 7, 9, 11]
        })
        
        item_id_to_name = {i: f"Item {i}" for i in range(1, 12)}
        id_to_history_part = {i: f"Item {i} [HistoryEmb]" for i in range(1, 12)}
        id_to_candidate_part = {i: f"Item {i} [CansEmb]" for i in range(1, 12)}
        
        return df, item_id_to_name, id_to_history_part, id_to_candidate_part

    def test_dataset_subset_indices(self, mock_data):
        df, item_id_to_name, id_to_history_part, id_to_candidate_part = mock_data
        
        # Define indices to select (e.g., indices 1 and 3)
        indices = [1, 3]
        
        dataset = SASRecDataset(
            df=df,
            max_seq_len=10,
            num_items=11,
            item_id_to_name=item_id_to_name,
            num_candidates=5,
            padding_item_id=0,
            id_to_history_part=id_to_history_part,
            id_to_candidate_part=id_to_candidate_part,
            indices=indices
        )
        
        # Check length
        assert len(dataset) == 2
        
        # Check content
        # Index 0 of dataset should be Index 1 of original df
        sample0 = dataset[0]
        assert sample0['next_item_id'] == 5 # Original index 1 has next_item 5
        
        # Index 1 of dataset should be Index 3 of original df
        sample1 = dataset[1]
        assert sample1['next_item_id'] == 9 # Original index 3 has next_item 9

    def test_dataset_no_indices(self, mock_data):
        df, item_id_to_name, id_to_history_part, id_to_candidate_part = mock_data
        
        dataset = SASRecDataset(
            df=df,
            max_seq_len=10,
            num_items=11,
            item_id_to_name=item_id_to_name,
            num_candidates=5,
            padding_item_id=0,
            id_to_history_part=id_to_history_part,
            id_to_candidate_part=id_to_candidate_part,
            indices=None
        )
        
        assert len(dataset) == 5
