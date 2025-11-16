# テストケース仕様書

このドキュメントは、本プロジェクトの各モジュールに対するテストケースを定義します。
**現状、すべてのテストがパスしており (`24 passed`)、プロジェクトの基本的な健全性が担保されています。**

---

## 1. `src/core` (完了: 11/11 passed)

### 1.1. `test_paths.py` (`src/core/paths.py` のテスト)

- **`test_get_project_root`**:
  - `get_project_root()` が返すパスが、このプロジェクトのルートディレクトリ（`v2`）を指していることを確認する。
  - 返されるパスが存在し、ディレクトリであることを確認する。

- **`test_get_data_dir`**:
  - `get_data_dir()` が返すパスが `(project_root)/data` であることを確認する。
  - 返されるパスが存在し、ディレクトリであることを確認する。

- **`test_get_result_dir`**:
  - `get_result_dir()` が返すパスが `(project_root)/result` であることを確認する。
  - 返されるパスが存在し、ディレクトリであることを確認する。

- **`test_get_conf_dir`**:
  - `get_conf_dir()` が返すパスが `(project_root)/conf` であることを確認する。
  - 返されるパスが存在し、ディレクトリであることを確認する。

- **`test_get_src_dir`**:
  - `get_src_dir()` が返すパスが `(project_root)/src` であることを確認する。
  - 返されるパスが存在し、ディレクトリであることを確認する。

### 1.2. `test_seed.py` (`src/core/seed.py` のテスト)

- **`test_set_seed_reproducibility`**:
  - `set_seed()` を同じシード値で2回呼び出した後、`random`, `numpy`, `torch` の乱数生成が同じ結果を返すことを確認する。
  - 異なるシード値で `set_seed()` を呼び出した後、乱数生成が異なる結果を返すことを確認する。

### 1.3. `test_logging.py` (`src/core/logging.py` のテスト)

- **`test_setup_logging_with_file`**:
  - `setup_logging()` をログディレクトリを指定して呼び出した後、ログファイルが作成されることを確認する。
  - ログファイルにメッセージが書き込まれることを確認する。
  - コンソールにもメッセージが出力されることを確認する（これは目視またはキャプチャで確認）。

- **`test_setup_logging_without_file`**:
  - `setup_logging()` をログディレクトリを指定せずに呼び出した後、ログファイルが作成されないことを確認する。
  - コンソールにはメッセージが出力されることを確認する。

### 1.4. `test_git_info.py` (`src/core/git_info.py` のテスト)

- **`test_get_git_info_in_repo`**:
  - Gitリポジトリ内で `get_git_info()` を呼び出した場合、`commit_hash` や `branch_name` などの情報が正しく取得できることを確認する。
  - 返される値が `None` ではないことを確認する。

- **`test_get_git_info_outside_repo`**:
  - Gitリポジトリ外（またはGitがインストールされていない環境）で `get_git_info()` を呼び出した場合、すべての値が `None` となることを確認する（要モック）。

### 1.5. `test_metrics.py` (`src/core/metrics.py` のテスト)

- **`test_calculate_metrics_perfect_match`**:
  - 予測が正解と完全に一致する場合、`recall@k`, `ndcg@k`, `hit_ratio@k` がすべて `1.0` になることを確認する。

- **`test_calculate_metrics_no_match`**:
  - 予測が正解と全く一致しない場合、すべてのメトリクスが `0.0` になることを確認する。

- **`test_calculate_metrics_partial_match`**:
  - 予測が部分的に一致する場合、メトリクスが期待される値になることを確認する（手計算で期待値を算出）。

- **`test_calculate_metrics_empty_input`**:
  - 予測リストや正解リストが空の場合、すべてのメトリクスが `0.0` になることを確認する。

- **`test_calculate_metrics_k_value`**:
  - `k` の値を変えることで、メトリクスの値が正しく変動することを確認する。

---

## 2. `src/student` (完了: 5/5 passed)

### 2.1. `test_models.py` (`src/student/models.py` のテスト)

- **`test_sasrec_forward_shape`**:
  - `SASRec` モデルの `forward` メソッドが、期待される形状 `(batch_size, hidden_size)` のテンソルを返すことを確認する。

- **`test_sasrec_predict_shape`**:
  - `SASRec` モデルの `predict` メソッドが、期待される形状 `(batch_size, num_items + 1)` のテンソルを返すことを確認する。

### 2.2. `test_datamodule.py` (`src/student/datamodule.py` のテスト)

- **`test_datamodule_setup`**:
  - `SASRecDataModule` の `setup` メソッドが、`train`, `val`, `test` の各データセットを正しく読み込めることを確認する。
  - `num_items` が正しく計算されていることを確認する。

- **`test_dataloader_batch_shape`**:
  - `train_dataloader` などが返すバッチの各要素（`item_seq`, `item_seq_len`, `next_item`）が期待される形状であることを確認する。

- **`test_dataloader_padding`**:
  - バッチ内の `item_seq` が、`max_seq_len` に合わせて正しくパディングされていることを確認する。

---

## 3. `src/teacher` (完了: 3/3 passed)

`iLoRAModel` の主要ロジックが実装されたことを受け、テストはダミー実装ではなく実際の動作を検証します。

### 3.1. `test_ilora_model.py` (`src/teacher/ilora_model.py` のテスト)

- **`test_ilora_forward_shape`**:
  - `iLoRAModel` の `forward` メソッドが、プロンプト変換、LoRAエキスパートの動的結合を経て、期待される形状 `(batch_size, num_items + 1)` の最終ロジットを返すことを確認する。

- **`test_ilora_get_teacher_outputs_shape`**:
  - `get_teacher_outputs` メソッドが返す辞書の各要素（`ranking_scores`, `embeddings`）が期待される形状であることを確認する。

- **`test_gating_network`**:
  - ゲーティングネットワークの出力が、形状 `(batch_size, num_lora_experts)` であり、softmaxを通過するため合計が1になることを確認する。

### 3.2. `test_trainer_ilora.py` (`src/teacher/trainer_ilora.py` のテスト)

- **`test_training_step`**:
  - `training_step` がスカラーの損失テンソルを返すことを確認する。
  - 損失が計算され、`nan` や `inf` にならないことを確認する。

---

## 4. `src/distill` (完了: 5/5 passed)

### 4.1. `test_kd_losses.py` (`src/distill/kd_losses.py` のテスト)

- **`test_ranking_distillation_loss`**:
  - 教師と生徒のロジットが同じ場合、ランキング蒸留損失が0に近くなることを確認する。
  - 温度 `T` を大きくすると、損失が小さくなることを確認する。

- **`test_embedding_distillation_loss`**:
  - 教師と生徒の埋め込みが同じ場合、埋め込み蒸留損失が0に近くなることを確認する（MSE, Cosine両方）。

### 4.2. `test_selection_policy.py` (`src/distill/selection_policy.py` のテスト)

- **`test_kl_divergence_policy_selection`**:
  - `KLDivergencePolicy` が、KLダイバージェンスが閾値を超えるサンプルを正しく選択することを確認する。
  - テストデータについて、`F.kl_div` の `log_target=True` 使用時のロジット処理を修正し、KLダイバージェンスが低いケースと高いケースの両方を適切に検証するように改善。
- **`test_ground_truth_error_policy_selection`**:
  - `GroundTruthErrorPolicy` が、生徒モデルの正解アイテムに対するロジットが特定の閾値を下回るサンプルを正しく選択することを確認する。

### 4.3. `test_trainer_distill.py` (`src/distill/trainer_distill.py` のテスト)

- **`test_distill_training_step`**:
  - `training_step` がスカラーの損失テンソルを返すことを確認する。
  - 各損失（ranking, embedding, ce）が正しく計算され、合計損失に反映されることを確認する。
  - 教師モデルのパラメータが学習中に更新されないことを確認する。
