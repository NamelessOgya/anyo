import pytest
import random
import numpy as np
import torch
from src.core.seed import set_seed

def test_set_seed_reproducibility():
    """
    set_seed() が乱数生成の再現性を保証するかテストします。
    """
    seed1 = 42
    seed2 = 123

    # --- シード1回目 ---
    set_seed(seed1)
    rand_val1_seed1 = random.random()
    np_val1_seed1 = np.random.rand()
    torch_val1_seed1 = torch.rand(1).item()

    # --- シード2回目 (同じシード) ---
    set_seed(seed1)
    rand_val2_seed1 = random.random()
    np_val2_seed1 = np.random.rand()
    torch_val2_seed1 = torch.rand(1).item()

    # 同じシードでは同じ乱数が生成されることを確認
    assert rand_val1_seed1 == rand_val2_seed1
    assert np_val1_seed1 == np_val2_seed1
    assert torch_val1_seed1 == torch_val2_seed1

    # --- 異なるシード ---
    set_seed(seed2)
    rand_val_seed2 = random.random()
    np_val_seed2 = np.random.rand()
    torch_val_seed2 = torch.rand(1).item()

    # 異なるシードでは異なる乱数が生成されることを確認
    assert rand_val1_seed1 != rand_val_seed2
    assert np_val1_seed1 != np_val_seed2
    assert torch_val1_seed1 != torch_val_seed2
