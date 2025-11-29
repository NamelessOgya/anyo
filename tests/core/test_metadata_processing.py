import unittest
import pandas as pd
from pathlib import Path
import tempfile
import os
from src.core.preprocess_data import process_metadata

class TestMetadataProcessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_process_ml100k_metadata(self):
        # Create dummy u.item
        # item_id | title | release date | video release date | IMDb URL | unknown | Action | ...
        content = "1|Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0\n" \
                  "2|GoldenEye (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?GoldenEye%20(1995)|0|1|1"
        
        with open(self.data_dir / "u.item", "w", encoding="latin-1") as f:
            f.write(content)

        df = process_metadata(self.data_dir, "ml-100k")
        
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['item_id'], 1)
        self.assertEqual(df.iloc[0]['title'], "Toy Story (1995)")
        self.assertEqual(df.iloc[1]['item_id'], 2)
        self.assertEqual(df.iloc[1]['title'], "GoldenEye (1995)")
        self.assertTrue('genres' in df.columns)

    def test_process_ml1m_metadata(self):
        # Create dummy movies.dat
        # item_id :: title :: genres
        content = "1::Toy Story (1995)::Animation|Children's|Comedy\n" \
                  "2::Jumanji (1995)::Adventure|Children's|Fantasy"
        
        with open(self.data_dir / "movies.dat", "w", encoding="latin-1") as f:
            f.write(content)

        df = process_metadata(self.data_dir, "ml-1m")
        
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['item_id'], 1)
        self.assertEqual(df.iloc[0]['title'], "Toy Story (1995)")
        self.assertEqual(df.iloc[0]['genres'], "Animation|Children's|Comedy")

if __name__ == '__main__':
    unittest.main()
