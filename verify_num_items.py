
import sys
import os
sys.path.append(os.getcwd())

from src.student.datamodule import SASRecDataModule
from transformers import AutoTokenizer

# Mock tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.add_special_tokens({'additional_special_tokens': ['[HistoryEmb]']})

dm = SASRecDataModule(
    dataset_name="ml-100k",
    data_dir="data/ml-100k",
    batch_size=4,
    max_seq_len=50,
    num_workers=0,
    tokenizer=tokenizer
)
dm.prepare_data()
dm.setup()

print(f"Num items: {dm.num_items}")
expected_items = 1682
if dm.num_items == expected_items:
    print(f"SUCCESS: num_items is {dm.num_items} as expected.")
else:
    print(f"FAILURE: num_items is {dm.num_items}, expected {expected_items}.")
