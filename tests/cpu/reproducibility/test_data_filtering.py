
import unittest
import pandas as pd
from src.student.datamodule import SASRecDataModule
from unittest.mock import MagicMock, patch
import shutil
import os

class TestDataFiltering(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data_filtering_tmp"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy movies.dat
        with open(f"{self.test_dir}/movies.dat", "w") as f:
            f.write("1::Toy Story::Animation\n")
            f.write("2::Jumanji::Adventure\n")
            f.write("3::Grumpier Old Men::Comedy\n")
            f.write("4::Waiting to Exhale::Comedy\n")
            f.write("5::Father of the Bride Part II::Comedy\n")

        # Create dummy train.csv with short and long sequences
        # User 1: seq len 2 (should be filtered)
        # User 2: seq len 3 (should be kept)
        # User 3: seq len 4 (should be kept)
        train_data = {
            "user_id": [1, 2, 3],
            "seq": ["1 2", "1 2 3", "1 2 3 4"],
            "next_item": [3, 4, 5]
        }
        pd.DataFrame(train_data).to_csv(f"{self.test_dir}/train.csv", index=False)
        
        # Create dummy val/test (can be empty or simple)
        pd.DataFrame(train_data).to_csv(f"{self.test_dir}/val.csv", index=False)
        pd.DataFrame(train_data).to_csv(f"{self.test_dir}/test.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_seq_len_filtering(self):
        dm = SASRecDataModule(
            dataset_name="dummy",
            data_dir=self.test_dir,
            batch_size=2,
            max_seq_len=10,
            num_workers=0
        )
        
        # Run setup which triggers loading and filtering
        dm.setup()
        
        # Check train_df
        # User 1 (index 0) should be gone.
        # User 2 (index 1) and User 3 (index 2) should remain.
        self.assertEqual(len(dm.train_df), 2)
        self.assertTrue(all(dm.train_df['seq'].apply(len) >= 3))
        
        # Check that user_id 1 is NOT present (assuming user_ids are preserved or remapped consistently)
        # The original user_id 1 had seq "1 2".
        # Let's check the content.
        seqs = dm.train_df['seq'].tolist()
        # seqs are now lists of mapped item IDs.
        # Original IDs 1, 2, 3 -> Mapped IDs (sorted unique)
        # Items: 1, 2, 3, 4, 5.
        # Mapped: 1->1, 2->2, 3->3, 4->4, 5->5 (since we have all in movies.dat)
        
        # Expected seqs: [1, 2, 3] (len 3), [1, 2, 3, 4] (len 4)
        # "1 2" (len 2) should be missing.
        
        lengths = [len(s) for s in seqs]
        self.assertIn(3, lengths)
        self.assertIn(4, lengths)
        self.assertNotIn(2, lengths)
        
        print("Data filtering test passed!")

if __name__ == "__main__":
    unittest.main()
