import random
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    :param seed: The integer seed value.
    """
    if seed is None:
        log.warning("Seed is None. Randomness will not be fixed.")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations for CuDNN if available
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log.info(f"Random seed set to {seed} for Python, NumPy, and PyTorch.")


if __name__ == "__main__":
    # Example usage
    print("Setting seed to 42...")
    set_seed(42)

    print("\nGenerating random numbers:")
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random (CPU): {torch.rand(1)}")
    if torch.cuda.is_available():
        print(f"PyTorch random (GPU): {torch.rand(1, device='cuda')}")

    print("\nSetting seed to 42 again and regenerating:")
    set_seed(42)
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random (CPU): {torch.rand(1)}")
    if torch.cuda.is_available():
        print(f"PyTorch random (GPU): {torch.rand(1, device='cuda')}")
