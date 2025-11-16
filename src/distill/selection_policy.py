from abc import ABC, abstractmethod
import torch
from omegaconf import DictConfig
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


class ISelectionPolicy(ABC):
    """
    Abstract base class for selecting samples for distillation.
    """

    @abstractmethod
    def select(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_info: Dict[str, Any],
        student_info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Selects which samples in a batch should have distillation loss applied.

        Args:
            batch: The current batch of data from the DataLoader.
            teacher_info: Dictionary containing teacher-specific information for the batch
                          (e.g., rankings, confidences).
            student_info: Dictionary containing student-specific information for the batch
                          (e.g., logits, embeddings, potentially loss/entropy).

        Returns:
            A boolean tensor of shape (batch_size,) where True indicates the sample is selected
            for distillation, and False otherwise.
        """
        pass


class SelectionPolicyAll(ISelectionPolicy):
    """
    A selection policy that selects all samples in a batch for distillation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        log.info(f"Initialized SelectionPolicyAll with strategy: {cfg.active.strategy}")

    def select(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_info: Dict[str, Any],
        student_info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Selects all samples for distillation.
        """
        batch_size = batch["seq"].shape[0]
        return torch.ones(batch_size, dtype=torch.bool, device=batch["seq"].device)


def get_selection_policy(cfg: DictConfig) -> ISelectionPolicy:
    """
    Factory function to get a selection policy instance based on configuration.
    """
    policy_name = cfg.active.strategy
    if policy_name == "all":
        return SelectionPolicyAll(cfg)
    # Future policies:
    # elif policy_name == "random":
    #     return SelectionPolicyRandom(cfg)
    # elif policy_name == "meta":
    #     return SelectionPolicyMeta(cfg)
    else:
        raise ValueError(f"Unknown selection policy: {policy_name}")


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf

    # Dummy config
    dummy_cfg = OmegaConf.create(
        {"active": {"name": "all", "strategy": "all", "budget_ratio": 1.0}}
    )

    # Get policy
    policy = get_selection_policy(dummy_cfg)
    print(f"Policy type: {type(policy)}")

    # Dummy batch data
    dummy_batch = {
        "seq": torch.randn(5, 10),
        "len_seq": torch.randint(1, 10, (5,)),
        "target": torch.randint(0, 100, (5,)),
        "original_index": torch.arange(5),
    }
    dummy_teacher_info = {}
    dummy_student_info = {}

    # Select samples
    selection_mask = policy.select(dummy_batch, dummy_teacher_info, dummy_student_info)
    print(f"Selection mask: {selection_mask}")
    print(f"Selection mask shape: {selection_mask.shape}")
    assert selection_mask.shape[0] == dummy_batch["seq"].shape[0]
    assert torch.all(selection_mask)  # For SelectionPolicyAll, all should be True

    # Test with unknown policy
    dummy_cfg.active.strategy = "random"
    try:
        get_selection_policy(dummy_cfg)
    except ValueError as e:
        print(f"Caught expected error for unknown policy: {e}")
