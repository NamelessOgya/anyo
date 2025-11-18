import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TeacherOutputDataset(Dataset):
    def __init__(self, teacher_outputs_batches_dir: Path):
        self.teacher_outputs_batches_dir = teacher_outputs_batches_dir
        self.file_paths = sorted(list(teacher_outputs_batches_dir.glob("batch_*.pt")))
        if not self.file_paths:
            raise ValueError(f"No teacher output batch files found in {teacher_outputs_batches_dir}")
        logger.info(f"Found {len(self.file_paths)} teacher output batch files.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            # Load the batch data
            batch_data = torch.load(file_path, map_location='cpu')
            return batch_data
        except Exception as e:
            logger.error(f"Error loading teacher output batch from {file_path}: {e}")
            # Depending on the error, you might want to return a dummy batch or re-raise
            raise

def teacher_output_collate_fn(batch):
    # Assuming each item in batch is a dictionary with 'ranking_scores', 'embeddings', etc.
    # And these are already batched tensors from the original teacher output generation
    # We just need to return them as is, or potentially stack them if they are single items
    
    # If the batch contains multiple dictionaries (e.g., if DataLoader batch_size > 1),
    # we need to concatenate them. However, our TeacherOutputDataset already returns
    # a full batch from a single file. So, the DataLoader's batch_size should be 1.
    # If batch_size is 1, 'batch' will be a list containing one dictionary.
    if len(batch) == 1:
        return batch[0]
    else:
        # This case should ideally not happen if DataLoader batch_size is 1
        # But if it does, we concatenate the tensors
        return {key: torch.cat([d[key] for d in batch], dim=0) for key in batch[0]}

