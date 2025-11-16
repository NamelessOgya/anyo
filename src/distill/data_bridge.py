import torch
import numpy as np
from omegaconf import DictConfig
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)


class DistillationDataBridge:
    """
    Reads teacher outputs (embeddings, rankings, confidences) and provides them
    in a format suitable for the distillation process.
    """

    def __init__(
        self, cfg: DictConfig, teacher_outputs_dir: Path, device: torch.device
    ):
        self.cfg = cfg
        self.teacher_outputs_dir = teacher_outputs_dir
        self.device = device
        self.llm_all_embeddings = None
        self.teacher_rankings = None
        self.teacher_confidences = None

    def load_teacher_outputs(self):
        """
        Loads all necessary teacher output files.
        """
        log.info(f"Loading teacher outputs from {self.teacher_outputs_dir}...")

        # Load all_embeddings.pt
        embeddings_path = self.teacher_outputs_dir / "all_embeddings.pt"
        if embeddings_path.exists():
            self.llm_all_embeddings = torch.load(embeddings_path).to(self.device)
            log.info(
                f"Loaded LLM all embeddings with shape: {self.llm_all_embeddings.shape}"
            )
        else:
            log.warning(
                f"LLM all embeddings not found at {embeddings_path}. Embedding distillation might be affected."
            )
            self.llm_all_embeddings = None  # Ensure it's None if not found

        # Load myrank_train.txt
        rankings_path = self.teacher_outputs_dir / "myrank_train.txt"
        if rankings_path.exists():
            # Assuming space-separated values, no header
            self.teacher_rankings = torch.LongTensor(np.loadtxt(rankings_path)).to(
                self.device
            )
            log.info(
                f"Loaded teacher rankings with shape: {self.teacher_rankings.shape}"
            )
        else:
            log.warning(
                f"Teacher rankings not found at {rankings_path}. Ranking distillation might be affected."
            )
            self.teacher_rankings = None

        # Load confidence_train.txt
        confidences_path = self.teacher_outputs_dir / "confidence_train.txt"
        if confidences_path.exists():
            # Assuming space-separated values, no header
            self.teacher_confidences = torch.FloatTensor(
                np.loadtxt(confidences_path)
            ).to(self.device)
            log.info(
                f"Loaded teacher confidences with shape: {self.teacher_confidences.shape}"
            )
        else:
            log.warning(
                f"Teacher confidences not found at {confidences_path}. Ranking distillation might be affected."
            )
            self.teacher_confidences = None

    def get_batch_teacher_info(self, original_indices: torch.Tensor) -> Dict[str, Any]:
        """
        Retrieves teacher-specific information for a given batch of original indices.
        """
        batch_info = {}
        if self.teacher_rankings is not None:
            batch_info["teacher_rankings"] = self.teacher_rankings[original_indices]
        if self.teacher_confidences is not None:
            batch_info["teacher_confidences"] = self.teacher_confidences[
                original_indices
            ]

        # llm_all_embeddings is used directly in the model forward pass, not per batch here.
        # It's passed to the evaluator/trainer directly.

        return batch_info


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    from src.core.logging import setup_logging
    import shutil

    # Setup logging
    setup_logging(Path("temp_log_data_bridge.log"))

    # Dummy config
    dummy_cfg = OmegaConf.create(
        {
            "distill": {"candidate_topk": 10},  # Used for dummy data generation
            "dataset": {"name": "test_dataset"},
        }
    )

    # Create dummy teacher output files
    temp_teacher_outputs_dir = Path("temp_teacher_outputs")
    temp_teacher_outputs_dir.mkdir(parents=True, exist_ok=True)

    num_items = 1000
    llm_emb_dim = 4096
    num_train_samples = 100
    top_n = dummy_cfg.distill.candidate_topk

    dummy_llm_all_embeddings = torch.randn(num_items, llm_emb_dim)
    dummy_teacher_rankings = torch.randint(0, num_items, (num_train_samples, top_n))
    dummy_teacher_confidences = torch.rand(num_train_samples, top_n)

    torch.save(dummy_llm_all_embeddings, temp_teacher_outputs_dir / "all_embeddings.pt")
    np.savetxt(
        temp_teacher_outputs_dir / "myrank_train.txt",
        dummy_teacher_rankings.numpy(),
        fmt="%d",
    )
    np.savetxt(
        temp_teacher_outputs_dir / "confidence_train.txt",
        dummy_teacher_confidences.numpy(),
        fmt="%.6f",
    )

    device = torch.device("cpu")  # Use CPU for example
    data_bridge = DistillationDataBridge(dummy_cfg, temp_teacher_outputs_dir, device)
    data_bridge.load_teacher_outputs()

    # Simulate getting batch info
    dummy_original_indices = torch.tensor([0, 5, 99], device=device)
    batch_teacher_info = data_bridge.get_batch_teacher_info(dummy_original_indices)

    print("\n--- Loaded Teacher Outputs ---")
    print(f"LLM All Embeddings shape: {data_bridge.llm_all_embeddings.shape}")
    print(f"Teacher Rankings shape: {data_bridge.teacher_rankings.shape}")
    print(f"Teacher Confidences shape: {data_bridge.teacher_confidences.shape}")

    print("\n--- Batch Teacher Info ---")
    print(
        f"Batch Teacher Rankings shape: {batch_teacher_info['teacher_rankings'].shape}"
    )
    print(
        f"Batch Teacher Confidences shape: {batch_teacher_info['teacher_confidences'].shape}"
    )

    # Clean up
    shutil.rmtree(temp_teacher_outputs_dir, ignore_errors=True)
    Path("temp_log_data_bridge.log").unlink(missing_ok=True)
