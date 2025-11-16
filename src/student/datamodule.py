import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from omegaconf import DictConfig
import logging
import os
import ast  # For safely evaluating string representations of lists/tuples

from src.core.data_utils import load_processed_data

log = logging.getLogger(__name__)


class SequentialRecommendationDataset(Dataset):
    """
    A custom Dataset for sequential recommendation tasks.
    Handles sequences of items and the next item to predict.
    """

    def __init__(self, dataframe: pd.DataFrame, max_seq_len: int, item_num: int):
        self.dataframe = dataframe
        self.max_seq_len = max_seq_len
        self.item_num = item_num  # Used for padding index

        # Pre-process sequences if they are stored as strings
        if isinstance(self.dataframe["seq"].iloc[0], str):
            self.dataframe["seq"] = self.dataframe["seq"].apply(ast.literal_eval)
        if isinstance(self.dataframe["next"].iloc[0], str):
            self.dataframe["next"] = self.dataframe["next"].apply(ast.literal_eval)
            # If 'next' is a list, take the first element (assuming single next item prediction)
            self.dataframe["next"] = self.dataframe["next"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Sequence of items
        seq = row["seq"]
        len_seq = row["len_seq"]
        target_item = row["next"]

        # Pad sequence to max_seq_len
        # Use item_num as padding_idx
        padded_seq = seq + [self.item_num] * (self.max_seq_len - len(seq))
        padded_seq = padded_seq[: self.max_seq_len]  # Ensure it's exactly max_seq_len

        return {
            "seq": torch.LongTensor(padded_seq),
            "len_seq": torch.LongTensor([len_seq]),  # Keep as tensor for consistency
            "target": torch.LongTensor([target_item]),
            "original_index": torch.LongTensor(
                [idx]
            ),  # To map back to original data for DLLM2Rec
        }


class StudentDataModule:
    """
    Manages data loading for student models, providing train, validation, and test DataLoaders.
    """

    def __init__(self, cfg: DictConfig, data_dir: str, device: torch.device):
        self.cfg = cfg
        self.data_dir = data_dir
        self.device = device
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.data_statis = None
        self.item_num = None
        self.max_seq_len = None

    def prepare_data(self):
        """
        Loads processed data using core.data_utils.
        """
        log.info("Preparing student data...")
        self.train_df, self.val_df, self.test_df, self.data_statis = (
            load_processed_data(self.cfg, self.data_dir)
        )
        self.item_num = self.data_statis["item_num"][0]
        self.max_seq_len = self.data_statis["seq_size"][
            0
        ]  # Assuming seq_size is max_seq_len

    def setup(self, stage: str = None):
        """
        Sets up datasets for train, validation, and test stages.
        """
        if self.train_df is None:  # Ensure data is prepared
            self.prepare_data()

        if stage == "fit" or stage is None:
            self.train_dataset = SequentialRecommendationDataset(
                self.train_df, self.max_seq_len, self.item_num
            )
            self.val_dataset = SequentialRecommendationDataset(
                self.val_df, self.max_seq_len, self.item_num
            )
            log.info(f"Train dataset size: {len(self.train_dataset)}")
            log.info(f"Validation dataset size: {len(self.val_dataset)}")
        if stage == "test" or stage is None:
            self.test_dataset = SequentialRecommendationDataset(
                self.test_df, self.max_seq_len, self.item_num
            )
            log.info(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.student.batch_size,
            shuffle=True,
            num_workers=0,  # For simplicity, can be configured
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.student.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.student.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )


if __name__ == "__main__":
    # Example usage with a dummy config
    from omegaconf import OmegaConf
    from src.core.paths import get_data_dir
    from src.core.data_utils import preprocess_data

    logging.basicConfig(level=logging.INFO)

    dummy_cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "test_dataset",
                "max_seq_len": 50,
                "min_user_inter": 5,
                "split": {"method": "leave_one_out"},
            },
            "paths": {"data_dir": "temp_data"},
            "student": {"batch_size": 32},
        }
    )

    temp_data_dir = get_data_dir(dummy_cfg)
    os.makedirs(temp_data_dir, exist_ok=True)
    preprocess_data(dummy_cfg, temp_data_dir)  # Ensure dummy data exists

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = StudentDataModule(dummy_cfg, temp_data_dir, device)
    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"\nItem num: {data_module.item_num}")
    print(f"Max seq len: {data_module.max_seq_len}")

    print("\n--- Sample from Train DataLoader ---")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  seq shape: {batch['seq'].shape}")
        print(f"  len_seq shape: {batch['len_seq'].shape}")
        print(f"  target shape: {batch['target'].shape}")
        print(f"  original_index shape: {batch['original_index'].shape}")
        if i == 0:
            break

    # Clean up
    import shutil

    shutil.rmtree(temp_data_dir)
    print(f"\nCleaned up {temp_data_dir}")
