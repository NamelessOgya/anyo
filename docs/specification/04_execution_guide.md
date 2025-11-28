# 実行ガイド

このドキュメントは、本プロジェクトで定義されている各実験を実行するための手順を説明します。
すべてのコマンドは、`ilora-dev-container` コンテナ内で実行することを前提としています。

---

## 1. 準備

### 1.1. コンテナイメージのビルド

実験環境を構築するために、まずDockerイメージをビルドします。
プロジェクトのルートディレクトリで以下のコマンドを実行してください。

```bash
docker build -t ilora-dllm2rec:latest .
```

### 1.2. コンテナの起動と接続

まず、以下のコマンドでコンテナを起動します。初回起動時は依存関係のインストールに時間がかかります。

```bash
# コンテナをバックグラウンドで起動
docker run -d --name ilora-dev-container -v "$(pwd)":/workspace -w /workspace --gpus all -it ilora-dllm2rec:latest

# 依存関係のインストール (初回のみ)
docker exec ilora-dev-container bash -c "poetry lock && poetry install"
```

以降、コンテナ内でコマンドを実行するには `docker exec` を使用します。
コンテナ内のシェルに入る場合は、以下のコマンドを実行します。

```bash
docker exec -it ilora-dev-container bash
```

### 1.3. 設定ファイルの確認

すべての実験は、`conf/` ディレクトリ以下の設定ファイルに基づいて実行されます。
メインの設定ファイルは `conf/config.yaml` であり、ここから各モジュールの設定が読み込まれます。

実験の挙動を変更したい場合は、`conf/` 配下の各 `.yaml` ファイルを修正するか、コマンドライン引数で上書きします。
特に、`train`コンフィグは`conf/config.yaml`で`train: default`が設定されているため、各実験タイプに対応する`train`コンフィグ（例: `train/student.yaml`, `train/teacher.yaml`, `train/distill.yaml`）をコマンドライン引数で明示的に指定する必要があります。

**データセットの行数制限 (`limit_data_rows`)**:
`conf/dataset/*.yaml` ファイルの `limit_data_rows` パラメータを使用すると、データセットから読み込む行数を制限できます。これは、開発やデバッグ時に迅速な実験を行うために非常に有用です。`src/exp/run_teacher.py` および `src/exp/run_distill.py` の両方でこの設定が適用されます。

**例：バッチサイズを変更して生徒モデルのベースラインを学習**
```bash
poetry run python -m src.exp.run_student_baseline train=student train.batch_size=64
```

### 1.4. データセットの準備

本プロジェクトでは MovieLens 1M データセットを使用します。
以下のスクリプトを実行することで、データセットのダウンロードと展開が自動的に行われます。

```bash
./cmd/movielens.sh
```

このスクリプトは、`data/ml-1m/` ディレクトリに `ratings.dat` などのデータファイルを配置します。

**注意:** `conf/dataset/movielens.yaml` ファイル内の `data_dir` が、上記で展開されたパス (`data/ml-1m`) を正しく指していることを確認してください。

---

## 2. テストの実行

`pytest` を使用して単体テストを実行します。`src` ディレクトリ内のモジュールを正しくインポートするため、`PYTHONPATH` の設定が必須です。

**コマンド:**
```bash
# PYTHONPATHを設定して全テストを実行
PYTHONPATH=/workspace poetry run pytest

# 個別のテストファイルを指定して実行
PYTHONPATH=/workspace poetry run pytest tests/teacher/test_ilora_model.py
```

**補足: 個別モジュールの自己テスト実行**
各Pythonモジュール内の `if __name__ == "__main__":` ブロックに記述された自己テストを実行する場合も、同様に `PYTHONPATH` を設定し、`-m` オプションでモジュールとして実行することを推奨します。

**例: `src/teacher/ilora_model.py` の自己テストを実行する場合**
```bash
docker exec ilora-dev-container bash -c "PYTHONPATH=/workspace poetry run python -m src.teacher.ilora_model"
```

**期待される出力:**
すべてのテストが `passed` となることを確認してください。`failed` や `error` が発生した場合は、実装やテストコードに問題がないか確認が必要です。

---

## 3. 実験の実行

### 3.1. 生徒モデルのベースライン学習

`SASRec` モデルを単体で学習・評価します。

**コマンド:**
```bash
# poetry run を使用する場合
poetry run python -m src.exp.run_student_baseline train=student

# または、cmd/ ディレクトリのスクリプトを使用する場合
./cmd/run_student_baseline.sh
```

**出力:**
- 学習ログ、TensorBoardログ、チェックポイント（ベストモデル、最終モデル）が `result/result_{timestamp}/` ディレクトリに保存されます。
- テストセットに対する評価結果が `result/result_{timestamp}/test_metrics.txt` に保存されます。

### 3.2. 教師モデル (iLoRA) の学習

`iLoRA` モデルを学習・評価します。

**注意:**
*   iLoRAモデルは大規模なLLMを使用するため、十分なGPUメモリが必要です。`torch.OutOfMemoryError` が発生した場合は、以下の点を確認してください。
    *   `conf/teacher/ilora.yaml` の `hidden_size` を小さくする。
    *   `conf/teacher/ilora.yaml` の `llm_model_name` をより小さなモデル（例: `facebook/opt-125m` はOPTモデルで最小）に変更する。
    *   `src/teacher/ilora_model.py` 内の `item_embeddings` 層の次元が適切に設定されているか確認する（`llm.config.vocab_size` ではなく `hidden_size` を使用し、プロジェクション層を介してLLMの入力次元に合わせる）。
*   **事前に学習済みのSASRecモデルのチェックポイントパスを指定する必要があります。** これは `conf/teacher/ilora.yaml` の `rec_model_checkpoint_path` で設定するか、コマンドライン引数で上書きします。

**コマンド:**
```bash
# poetry run を使用する場合
# 例: poetry run python -m src.exp.run_teacher train=teacher teacher.rec_model_checkpoint_path=/path/to/your/sasrec_checkpoint.ckpt
poetry run python -m src.exp.run_teacher train=teacher

# または、cmd/ ディレクトリのスクリプトを使用する場合
./cmd/run_teacher.sh
```

**出力:**
- 学習ログ、TensorBoardログ、チェックポイントが `result/result_{timestamp}/` ディレクトリに保存されます。
- テストセットに対する評価結果がコンソールに出力されます。

#### 3.2.1. Teacherモデルの高度な設定 (2025-11-27 追加)

以下のパラメータを `conf/teacher/ilora.yaml` またはコマンドライン引数で設定することで、学習挙動を詳細に制御できます。

**1. Student Embeddingの利用設定**
*   `use_item_embeddings_head` (bool, default: `True`):
    *   `True`: StudentのItem EmbeddingをHeadとして使用し、Reverse Distillationを行います。
    *   `False`: ランダム初期化されたLinear Headを使用し、Student Embeddingは使用しません（従来手法）。

**2. Embedding Imitation (Reverse Distillation) のLoss設定**
*   `distill_loss_type` (str, default: `mse`):
    *   `mse`: 平均二乗誤差。ベクトルの大きさと方向の両方を近づけます。
    *   `cosine`: コサイン類似度。ベクトルの方向のみを近づけます。
    *   `l1`: 絶対値誤差。外れ値にロバストです。
    *   `huber`: MSEとL1のハイブリッド。
    *   `contrastive`: InfoNCE形式。正例ペアの類似度を上げ、負例ペアを下げます。

**3. 正則化強度の減衰 (Decay) 設定**
*   `distill_decay_type` (str, default: `linear`):
    *   `none`: 減衰なし。常に `distill_lambda` の値を使用します。
    *   `linear`: 直線的に減衰します。
    *   `cosine`: コサインカーブに従って減衰します。
    *   `exponential`: 指数関数的に減衰します。
*   `distill_min_lambda` (float, default: `0.0`): 減衰後の最小値。
*   `distill_decay_steps` (int, default: `null`):
    *   減衰が完了するまでのステップ数。
    *   `null` の場合は全学習ステップ (`max_steps`) かけて減衰します。
    *   例: `1000` に設定すると、最初の1000ステップで `distill_min_lambda` まで減衰し、以降は維持されます。

**コマンド例:**
```bash
# Cosine Lossを使用し、最初の100ステップで正則化を0にする設定
poetry run python -m src.exp.run_teacher \
    train=teacher \
    teacher.use_item_embeddings_head=True \
    teacher.distill_loss_type=cosine \
    teacher.distill_decay_type=linear \
    teacher.distill_min_lambda=0.0 \
    teacher.distill_decay_steps=100
```

### 3.3. 知識蒸留

学習済みの教師モデルを用いて、生徒モデルに知識を蒸留します。

**コマンド:**
```bash
# poetry run を使用する場合
poetry run python -m src.exp.run_distill train=distill

# または、cmd/ ディレクトリのスクリプトを使用する場合
./cmd/run_distill.sh
```
**注意:** このスクリプトを実行する前に、`run_teacher.py` を実行して教師モデルを学習し、生成されたチェックポイントパスを `conf/distill/dllm2rec.yaml` の `distill.teacher_checkpoint_path` に設定しておく必要があります。現在、このパスは以前に学習された教師モデルのチェックポイントで埋められています。

**出力:**
- 蒸留学習のログ、TensorBoardログ、チェックポイントが `result/result_{timestamp}/` ディレクトリに保存されます。
- 蒸留後の生徒モデルの評価結果が `result/result_{timestamp}/test_metrics.txt` に保存されます。

### 3.4. 全モデルの評価

複数の学習済みモデル（ベースライン、蒸留済み、教師）をまとめて評価します。

**コマンド:**
```bash
# poetry run を使用する場合
poetry run python -m src.exp.run_eval_all

# または、cmd/ ディレクトリのスクリプトを使用する場合
./cmd/run_eval_all.sh
```
**注意:** このスクリプトを実行する前に、`conf/eval/default.yaml` またはコマンドライン引数で、評価対象となる各モデルのチェックポイントパスを指定する必要があります。

**出力:**
- 各モデルの評価結果がコンソールに出力されます。
- すべての評価結果をまとめたJSONファイルが `result/result_{timestamp}/all_evaluation_results.json` に保存されます。

---

## 4. Google Colab での実行

Google Colab などの非コンテナ環境で実行する場合の手順です。

### 4.1. 環境構築

1.  **Poetryのインストール**:
    ```bash
    bash cmd/colab/00_install_poetry.sh
    ```

2.  **依存関係のインストール**:
    ```bash
    bash cmd/colab/01_install_dependencies.sh
    ```
    ※ `SentencePiece` エラーが出た場合は `poetry add sentencepiece` を実行後に再試行してください。

3.  **Hugging Face 認証**:
    LLMモデル（Qwen/OPT等）のダウンロードに必要です。
    ```bash
    python authenticate_hf.py
    ```

### 4.2. 実験の実行（Colab用スクリプト）

`cmd/colab/` 配下のスクリプトを使用します。これらは依存関係のパス解決などを自動で行います。

*   **データ準備**:
    ```bash
    bash cmd/colab/02_prepare_dataset.sh
    ```

*   **生徒モデル（SASRec）学習**:
    ```bash
    bash cmd/colab/10_run_student_baseline.sh
    ```

*   **教師モデル（iLoRA）学習**:
    ```bash
    # 生徒モデルのチェックポイントパスは引数で指定可能です（任意）
    bash cmd/colab/11_run_teacher.sh
    ```

*   **知識蒸留**:
    ```bash
    bash cmd/colab/12_run_distill.sh
    ```

---

## 5. Flash Attention Best Practices

A100/L4 GPU環境で Flash Attention 2 を有効化するためのガイドです。

### 5.1. Google Colab / 非コンテナ環境

`pip install flash-attn` はビルドに時間がかかるため、**Pre-built Wheel** の使用を推奨します。

1.  **環境確認**:
    ```python
    import torch
    print(torch.__version__)       # 例: 2.4.0+cu121
    print(torch.version.cuda)      # 例: 12.1
    ```

2.  **インストール**:
    [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) から環境に合うWheelを探してインストールします。
    ```bash
    # 例: PyTorch 2.4, CUDA 12.3, Python 3.10
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    ```

### 5.2. トラブルシューティング

*   **"undefined symbol" エラー**:
    *   PyTorchとFlash Attentionのビルド環境（GLIBC, cxx11abi）の不一致が原因です。別のWheelを試すか、ソースビルドを行ってください。
*   **"CUDA capability sm_xx is not compatible"**:
    *   T4 GPUなどはFlash Attention 2に対応していません。A100またはL4を使用してください。
