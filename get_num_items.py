import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pytorch_lightning as pl

from src.core.paths import get_project_root
from src.core.seed import set_seed
from src.core.logging import setup_logging

from src.student.datamodule import SASRecDataModule

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def get_num_items(cfg: DictConfig):
    dm = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.student.max_seq_len,
        num_workers=cfg.train.num_workers,
        limit_data_rows=cfg.dataset.limit_data_rows
    )
    dm.prepare_data()
    dm.setup()
    print(f"Number of items: {dm.num_items}")

if __name__ == "__main__":
    get_num_items()
