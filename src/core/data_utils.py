import os
import pandas as pd
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


def preprocess_data(cfg: DictConfig, data_dir: str):
    """
    Generates processed data from raw data based on dataset configuration.
    This is a placeholder function. Actual implementation will depend on
    the specific dataset and the requirements of iLoRA/DLLM2Rec.
    """
    dataset_name = cfg.dataset.name
    log.info(f"Preprocessing data for dataset: {dataset_name}")

    # Example: Check if processed data already exists
    processed_data_path = os.path.join(data_dir, f"{dataset_name}_processed.pkl")
    if os.path.exists(processed_data_path):
        log.info(
            f"Processed data already exists at {processed_data_path}. Skipping preprocessing."
        )
        return

    log.info(f"Starting data preprocessing for {dataset_name}...")
    # --- Placeholder for actual data processing logic ---
    # This might involve:
    # 1. Reading raw data (e.g., CSV files)
    # 2. Filtering users/items based on interaction counts (min_user_inter)
    # 3. Mapping original IDs to sequential IDs
    # 4. Creating sequential data (e.g., user history sequences)
    # 5. Splitting into train/validation/test sets
    # 6. Saving processed dataframes/files

    # For now, let's just create dummy files to simulate processing
    # In a real scenario, this would call specific processing scripts or functions
    # from iLoRA or DLLM2Rec's data utilities.

    # Dummy data creation
    dummy_train_df = pd.DataFrame({"seq": [[1, 2, 3]], "len_seq": [3], "next": [4]})
    dummy_val_df = pd.DataFrame({"seq": [[1, 2, 3, 4]], "len_seq": [4], "next": [5]})
    dummy_test_df = pd.DataFrame(
        {"seq": [[1, 2, 3, 4, 5]], "len_seq": [5], "next": [6]}
    )
    dummy_data_statis_df = pd.DataFrame(
        {"seq_size": [cfg.dataset.max_seq_len], "item_num": [1000]}
    )

    os.makedirs(os.path.join(data_dir, dataset_name), exist_ok=True)
    dummy_train_df.to_pickle(os.path.join(data_dir, dataset_name, "train_data.df"))
    dummy_val_df.to_csv(
        os.path.join(data_dir, dataset_name, "val_data.csv"), index=False
    )
    dummy_test_df.to_csv(
        os.path.join(data_dir, dataset_name, "test_data.csv"), index=False
    )
    dummy_data_statis_df.to_pickle(
        os.path.join(data_dir, dataset_name, "data_statis.df")
    )

    log.info(f"Data preprocessing for {dataset_name} completed (dummy files created).")


def load_processed_data(cfg: DictConfig, data_dir: str):
    """
    Loads processed train, validation, and test data.
    """
    dataset_name = cfg.dataset.name
    dataset_path = os.path.join(data_dir, dataset_name)

    train_df = pd.read_pickle(os.path.join(dataset_path, "train_data.df"))
    val_df = pd.read_csv(os.path.join(dataset_path, "val_data.csv"))
    test_df = pd.read_csv(os.path.join(dataset_path, "test_data.csv"))
    data_statis_df = pd.read_pickle(os.path.join(dataset_path, "data_statis.df"))

    log.info(f"Loaded processed data for {dataset_name}.")
    return train_df, val_df, test_df, data_statis_df


if __name__ == "__main__":
    # Example usage with a dummy config
    from omegaconf import OmegaConf

    dummy_cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "test_dataset",
                "max_seq_len": 50,
                "min_user_inter": 5,
                "split": {"method": "leave_one_out"},
            },
            "paths": {"data_dir": "temp_data"},
        }
    )

    temp_data_dir = dummy_cfg.paths.data_dir
    os.makedirs(temp_data_dir, exist_ok=True)

    preprocess_data(dummy_cfg, temp_data_dir)
    train, val, test, statis = load_processed_data(dummy_cfg, temp_data_dir)

    print("\n--- Loaded Data ---")
    print("Train head:\n", train.head())
    print("Val head:\n", val.head())
    print("Test head:\n", test.head())
    print("Data Statis:\n", statis)

    # Clean up
    import shutil

    shutil.rmtree(temp_data_dir)
    print(f"\nCleaned up {temp_data_dir}")
