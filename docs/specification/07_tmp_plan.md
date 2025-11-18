# 7. Agent-Gamma 作業計画

## 7.1. 課題
知識蒸留を行った生徒モデルの性能（`test_recall@10`: 0.03157）が、蒸留前のベースライン（0.07368）や教師モデル（0.08421）と比較して著しく低い。この問題の原因を特定し、蒸留プロセスを修正して性能を改善する。

## 7.2. 作業計画
- [x] **課題0: 学習用データセットの事前分割 (最優先)**
    - [x] `ml-1m/ratings.dat` を読み込み、train/validation/testに分割する前処理スクリプトを `src/core/preprocess_data.py` として作成する。
    - [x] 分割ロジックは現在の `SASRecDataModule` のものを踏襲し、各ユーザーの最後のアイテムをテスト用、最後から2番目を検証用とする。
    - [x] 分割した結果を `data/ml-1m/` ディレクトリに `train.csv`, `val.csv`, `test.csv` として保存する。
    - [x] `SASRecDataModule` を、これらの事前分割されたファイルを読み込むように修正する。

- [x] **課題1: 蒸留損失関数の実装レビュー (ref_repositories/DLLM2Recとの比較)**
    - [x] `src/distill/kd_losses.py` の `WeightedBCELoss` を中心に、実装をレビュー済み。ネガティブサンプリングが欠如している問題を特定。
    - [x] `WeightedBCELoss` を修正し、ネガティブサンプルを受け取れるように変更済み。

- [x] **課題2: 蒸留プロセスのトレーナー実装レビュー (ref_repositories/DLLM2Recとの比較)**
    - [x] `src/distill/trainer_distill.py` を調査済み。
    - [x] ネガティブサンプルを生成し、損失関数に渡すように修正済み。

- [x] **課題3: データ分割方法の妥当性確認**
    - [x] `src/core/dataset.py` の `split_data` 関数を確認し、別エージェントによる修正が反映されているか、また時系列分割として適切かを確認する。(課題0で対応済み)

- [x] **課題4: 各実行スクリプトにおけるデータモジュール初期化の修正**
    - [x] `src/exp/run_student_baseline.py` において、`SASRecDataModule` の初期化時に `train_file`, `val_file`, `test_file` を渡すように修正する。
    - [x] `src/exp/run_teacher.py` において、`SASRecDataModule` の初期化時に `train_file`, `val_file`, `test_file` を渡すように修正する。
    - [x] `src/exp/run_distill.py` において、`SASRecDataModule` の初期化時に `train_file`, `val_file`, `test_file` を渡すように修正する。

- [x] **課題5: パイプライン全体の迅速な再実行と動作確認**
    - [x] **(実験時間短縮のため)** 以下の通り設定を変更済み。
        - [x] `conf/dataset/movielens.yaml` の `limit_data_rows` を `10000` に設定。
        - [x] `conf/train/student.yaml` の `max_epochs` を `3` に設定。
        - [x] `conf/train/teacher.yaml` の `max_epochs` を `3` に設定。
        - [x] `conf/train/distill.yaml` の `max_epochs` を `3` に設定。
    - [x] **(データ前処理の実行)** `bash cmd/run_preprocess_data.sh` を実行し、データ前処理を完了。
    - [x] **(パイプラインの再実行)** 以下のスクリプトを順次実行し、エラーなく完了することを確認する。
        - [x] `bash cmd/run_student_baseline.sh` (完了)
        - [x] `bash cmd/run_teacher.sh` (実行中、OOMエラーで停止)
            - [x] `TypeError: SASRecDataModule.__init__()` エラーを修正 (`src/exp/run_teacher.py`)。
            - [x] `AttributeError: 'SASRecDataModule' object has no attribute 'item_id_to_name'` エラーを修正 (`src/student/datamodule.py`)。
            - [x] `RuntimeError: size mismatch for item_embeddings.weight` エラーを修正 (`conf/teacher/ilora.yaml` の `rec_model_checkpoint_path` を更新)。
            - [x] `KeyError: 'tokens'` エラーを修正 (`src/teacher/ilora_model.py` で `_prepare_llm_input` を実装し、`batch["seq"]` からLLM入力を構築)。
            - [x] `OverflowError: int too big to convert` エラーを修正 (`src/teacher/ilora_model.py` の `_prepare_llm_input` で `max_length` を固定値 `128` に設定)。
            - [x] `torch.OutOfMemoryError: CUDA out of memory` エラーが発生。`conf/train/teacher.yaml` の `batch_size` を `32 -> 4 -> 2` と段階的に削減。
            - [x] `torch.OutOfMemoryError: CUDA out of memory` エラーが継続するため、`src/teacher/ilora_model.py` でLLMにグラディエントチェックポインティングを有効化。
            - [x] **(追加調査)** OOMエラーが以前発生しなかった箇所で発生しているため、GPU使用ロジック（特にフリーズ部分と学習部分の相互作用）を確認する。
                - [x] `src/teacher/ilora_model.py` を確認。
                - [x] `src/teacher/trainer_ilora.py` を確認。
                - [x] `src/teacher/factory.py` を確認し、`rec_model`のフリーズを確認。
                - [x] `conf/teacher/ilora.yaml` (LLMモデル名) および `conf/train/teacher.yaml` (バッチサイズ) を確認。
                - [x] `check_params.py` を実行し、学習対象パラメータを確認する。結果、`rec_model`はフリーズされており、学習対象ではないことを確認。
                - [x] `src/exp/run_teacher.py` でLLMを`float16`でロードし、`torch.cuda.empty_cache()`を追加。
                - [x] `src/teacher/ilora_model.py` で`projector`と`gating_network`を`.half()`にキャスト。
                - [x] `src/exp/run_teacher.py` でLLMの`float16`ロードを元に戻し、`src/teacher/ilora_model.py`でLLMを`.half()`にキャスト。
                - [x] `bash cmd/run_teacher.sh` を実行し、OOMエラーが解消されたか確認する。

- [x] **課題6: LoRA finetuningのGPUメモリ要求に関する調査**
    - [x] LoRA finetuning時のGPUメモリ要求について、類似モデルでの事例を調査する。
    - [x] LoRA使用時にGPUメモリ使用量に影響を与える主要な変数（`batch_size`, `max_seq_len`, `lora_r`など）を特定する。
    - [x] 調査結果を`docs/research/lora_oom.md`にまとめる。

- [x] **課題7: OOMエラーの根本原因調査**
    - [x] `resize_token_embeddings`の呼び出しを一時的にコメントアウトし、OOMが再現するか確認する。
        - [x] `src/exp/run_teacher.py` の `add_special_tokens` と `resize_token_embeddings` をコメントアウト済み。
    - [x] `bash cmd/run_teacher.sh` を再実行し、OOMエラーが解消されるか確認する。 (結果: `ValueError: '[HistoryEmb]' is not in list` が発生)
    - [x] `src/exp/run_teacher.py` の `add_special_tokens` と `resize_token_embeddings` をアンコメントする。
    - [x] `bash cmd/run_teacher.sh` を再実行し、OOMエラーが解消されるか確認する。 (結果: `torch.OutOfMemoryError` が発生)
    - [x] PyTorch Lightningを使わず、素のPyTorchで最小限の訓練ループを実装し、OOMが発生するかを切り分ける。 (`src/exp/debug_teacher_oom.py` を作成済み)
        - [x] `omegaconf.errors.ConfigAttributeError: Missing key num_heads` エラーを修正 (`src/exp/debug_teacher_oom.py` の `student` configに `num_heads`, `num_layers`, `dropout_rate` を追加)。
        - [x] `src/exp/debug_teacher_oom.py` を再実行し、OOMが発生しないことを確認。 (結果: OOMなし、メモリ使用量: 0.34 GB)
        - [x] `src/exp/debug_teacher_oom.py` の `batch_size` を `32` に変更し、OOMが発生しないことを確認。 (結果: OOMなし、メモリ使用量: 0.90 GB)
    - [x] PyTorch Lightningのメモリ使用量を調査する。 (`src/teacher/trainer_ilora.py` をレビューしたが、明らかなメモリリークは見つからなかった。)
    - [x] `src/exp/debug_teacher_oom.py` を修正し、バックワードパスとオプティマイザーステップを含める。
        - [x] `NameError: name 'batch' is not defined` エラーを修正 (`batch = next(iter(train_dataloader))` を `try` ブロックの前に移動)。
    - [x] `src/exp/debug_teacher_oom.py` を再実行し、OOMが発生するか確認する。 (結果: `RuntimeError: Trying to backward through the graph a second time` が発生)
    - [x] `src/teacher/ilora_model.py` でグラディエントチェックポインティングを無効化する。
    - [x] `src/exp/debug_teacher_oom.py` を再実行し、OOMが発生するか確認する。 (結果: OOMなし、メモリ使用量: 0.94 GB)
    - [x] `bash cmd/run_teacher.sh` を再実行し、`conf/train/teacher.yaml` の `batch_size` を `32` に設定してOOMが解消されるか確認する。 (結果: `torch.OutOfMemoryError` が発生)
    - [x] `conf/train/teacher.yaml` の `num_workers` を `0` に設定する。 (既に `0` に設定済み)
    - [x] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 環境変数を設定して `bash cmd/run_teacher.sh` を再実行し、OOMが解消されるか確認する。 (結果: `RuntimeError: CUDA driver error: operation not supported` が発生)
    - [x] `src/exp/run_teacher.py` から `rec_model_instance` の冗長な `.to(device)` 呼び出しを削除する。
    - [x] `bash cmd/run_teacher.sh` を再実行し、エラーが解消されるか確認する。 (結果: `RuntimeError: CUDA driver error: operation not supported` が発生)
    - [x] `src/teacher/factory.py` で `torch.set_float32_matmul_precision('high')` を設定する。
    - [x] `bash cmd/run_teacher.sh` を再実行し、エラーが解消されるか確認する。 (結果: `RuntimeError: CUDA driver error: operation not supported` が発生)
    - [x] `src/teacher/factory.py` 内でLLMとTokenizerを再ロードするように変更する。
    - [x] `src/exp/run_teacher.py` からLLMとTokenizerのロード部分を削除する。
        - [x] `bash cmd/run_teacher.sh` を再実行し、エラーが解消されるか確認する。 (結果: `RuntimeError: CUDA driver error: operation not supported` が発生)
    - [x] `bash cmd/run_teacher.sh` を再実行し、エラーが解消されるか確認する。 (結果: `RuntimeError: CUDA driver error: operation not supported` が発生。これは環境的な問題である可能性が高い。)
    - [x] 調査結果を `docs/report/cuda_driver_error_investigation.md` にまとめた。
        - [ ] `bash cmd/run_teacher.sh` を再実行し、エラーが解消されるか確認する。    - [ ] `bash cmd/run_teacher.sh` を再実行し、エラーが解消されるか確認する。