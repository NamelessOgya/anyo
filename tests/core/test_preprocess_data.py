import unittest
import pandas as pd
import numpy as np
from src.core.preprocess_data import process_and_split

class TestPreprocessData(unittest.TestCase):
    def test_loo_splitting(self):
        # User 1: 5 items, sorted by time
        data = {
            'user_id': [1, 1, 1, 1, 1],
            'item_id': [101, 102, 103, 104, 105],
            'rating': [5, 5, 5, 5, 5],
            'timestamp': [10, 20, 30, 40, 50]
        }
        df = pd.DataFrame(data)
        
        train_df, val_df, test_df = process_and_split(df, min_seq_len=3)
        
        # Check Test (Last item)
        self.assertEqual(len(test_df), 1)
        self.assertEqual(test_df.iloc[0]['user_id'], 1)
        self.assertEqual(test_df.iloc[0]['next_item'], 105)
        self.assertEqual(test_df.iloc[0]['seq'], "101 102 103 104")
        
        # Check Val (2nd Last item)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(val_df.iloc[0]['user_id'], 1)
        self.assertEqual(val_df.iloc[0]['next_item'], 104)
        self.assertEqual(val_df.iloc[0]['seq'], "101 102 103")
        
        # Check Train (Remaining)
        # Sequence: 101, 102, 103, 104, 105
        # Train samples:
        # 1. seq=[101], next=102
        # 2. seq=[101, 102], next=103
        self.assertEqual(len(train_df), 2)
        
        row1 = train_df.iloc[0]
        self.assertEqual(row1['next_item'], 102)
        self.assertEqual(row1['seq'], "101")
        
        row2 = train_df.iloc[1]
        self.assertEqual(row2['next_item'], 103)
        self.assertEqual(row2['seq'], "101 102")

    def test_timestamp_sorting(self):
        # User 2: 4 items, unsorted timestamps
        # Items: A(100), B(10), C(50), D(5)
        # Expected Sorted: D(5), B(10), C(50), A(100)
        # IDs: 201(100), 202(10), 203(50), 204(5)
        # Sorted IDs: 204, 202, 203, 201
        data = {
            'user_id': [2, 2, 2, 2],
            'item_id': [201, 202, 203, 204],
            'rating': [5, 5, 5, 5],
            'timestamp': [100, 10, 50, 5]
        }
        df = pd.DataFrame(data)
        
        train_df, val_df, test_df = process_and_split(df, min_seq_len=3)
        
        # Check Test (Last = 201)
        self.assertEqual(len(test_df), 1)
        self.assertEqual(test_df.iloc[0]['next_item'], 201)
        # Seq should be 204, 202, 203
        self.assertEqual(test_df.iloc[0]['seq'], "204 202 203")
        
        # Check Val (2nd Last = 203)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(val_df.iloc[0]['next_item'], 203)
        # Seq should be 204, 202
        self.assertEqual(val_df.iloc[0]['seq'], "204 202")
        
        # Check Train
        # Sorted: 204, 202, 203, 201
        # Train samples:
        # 1. seq=[204], next=202
        self.assertEqual(len(train_df), 1)
        self.assertEqual(train_df.iloc[0]['next_item'], 202)
        self.assertEqual(train_df.iloc[0]['seq'], "204")

if __name__ == '__main__':
    unittest.main()
