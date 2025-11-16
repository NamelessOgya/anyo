import random
import numpy as np
import torch
import os

def set_seed(seed: int):
    """
    実験の再現性を確保するために、すべての乱数シードを設定します。

    Args:
        seed (int): 設定するシード値。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 決定論的アルゴリズムを有効にする
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Pythonのハッシュシードも設定
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    # テスト用の簡単な例
    test_seed = 42
    set_seed(test_seed)
    print(f"Seed set to {test_seed}")

    # 乱数生成器がシードされていることを確認
    print(f"Random int: {random.randint(0, 100)}")
    print(f"NumPy random: {np.random.rand()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA random: {torch.rand(1).item()}")
    else:
        print(f"PyTorch CPU random: {torch.rand(1).item()}")
