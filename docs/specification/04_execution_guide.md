# 実行ガイド

このドキュメントは、本プロジェクトで定義されている各実験を実行するための手順を説明します。
すべてのコマンドは、`ilora-dev-container` コンテナ内で実行することを前提としています。

---

## 1. 準備

### 1.1. コンテナの起動と接続

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

### 1.2. 設定ファイルの確認

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
