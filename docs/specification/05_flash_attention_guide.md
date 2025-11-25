# Flash Attention Best Practices (Google Colab & Docker)

## 1. Google Colab

Google Colab環境でFlash Attention (v2) を使用するためのベストプラクティスです。

### 1.1. GPU要件
*   **必須**: NVIDIA Ampere (A100), Ada (RTX 40xx), Hopper (H100) 世代のGPU。
*   **注意**: T4 (無料枠で一般的) はFlash Attention v2に対応していません（v1.xのみ）。Colab Pro/Pro+で **A100** または **L4** を選択する必要があります。

### 1.2. インストール方法 (Pre-built Wheel推奨)
`pip install flash-attn` を直接実行するとソースビルドが始まり、時間がかかる上に失敗することが多いため、**Pre-built Wheel** の使用を強く推奨します。

1.  **環境の確認**:
    ```python
    import torch
    print(torch.__version__)       # 例: 2.4.0+cu121
    print(torch.version.cuda)      # 例: 12.1
    !python --version              # 例: Python 3.10.12
    ```

2.  **Wheelの選択とインストール**:
    [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) から、環境に合致するWheelのURLを探します。
    
    *   命名規則: `flash_attn-{version}+cu{cuda}torch{torch}cxx11abi{abi}-cp{python}-...`
    *   例 (PyTorch 2.4, CUDA 12.3, Python 3.10, cxx11abi=FALSE):
    
    ```python
    !pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ```

## 2. Docker Environment

Docker環境で安定して動作させるための戦略です。

### 2.1. Base Imageの選定
PyTorch公式イメージまたはNVIDIA NGCイメージを使用するのが最も確実です。

*   **推奨**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` (今回使用中)
*   **代替**: `nvcr.io/nvidia/pytorch:24.07-py3` (NVIDIA最適化済み、ライブラリが豊富)

### 2.2. インストール戦略

#### 戦略A: Pre-built Wheel (最速)
Colabと同様に、環境に合ったWheelをインストールします。
*   **課題**: GLIBCのバージョンや `cxx11abi` の不一致により `ImportError` が発生することがある（今回のケース）。
*   **対策**: `torch._C._GLIBCXX_USE_CXX11_ABI` を確認し、合致するWheelを選ぶ。それでもダメな場合は戦略Bへ。

#### 戦略B: ソースビルド (確実)
Dockerビルド時にソースからコンパイルします。時間はかかりますが、その環境に完全に適合したバイナリが生成されます。

**Dockerfile記述例**:
```dockerfile
# ビルド依存関係
RUN pip install ninja packaging

# Flash Attentionのビルド (数十分かかります)
# MAX_JOBSを制限しないとメモリ不足になることがあります
ENV MAX_JOBS=4
RUN pip install flash-attn --no-build-isolation
```

#### 戦略C: プリインストール済みイメージの使用
Hugging FaceのDLC (Deep Learning Containers) や、`winglian/axolotl` などのLLM学習用イメージは、Flash Attentionがプリインストールされていることが多いです。これらをベースに使うと手間が省けます。

## 3. トラブルシューティング

### "undefined symbol" エラー
*   **原因**: PyTorchとFlash Attentionのビルドに使われたコンパイラやライブラリ（GLIBC, cxx11abi）の不一致。
*   **解決策**:
    1.  `cxx11abi` (TRUE/FALSE) が合っているか再確認。
    2.  解決しない場合は **ソースビルド (戦略B)** に切り替える。

### "CUDA capability sm_xx is not compatible"
*   **原因**: GPUアーキテクチャとFlash Attentionのビルド設定が合っていない（例: T4でFA2を使おうとした）。
*   **解決策**: GPUに合ったバージョンを使うか、対応GPU (A100等) を使う。
