# 7. Agent-Gamma 作業計画 (PyTorch Lightning不使用版)

## 7.1. 課題

教師モデルの学習時に発生する `RuntimeError: CUDA driver error: operation not supported` を解決し、知識蒸留パイプラインの正確な実行を可能にする。

## 7.2. 作業計画

### 7.2.1. `src/exp/run_teacher.py` の PyTorch Lightning 依存の排除

*   [x] `src/exp/run_teacher.py` から `pytorch_lightning` のインポートおよび関連コードを削除する。
*   [x] `pl.Trainer` のインスタンス化と使用を削除する。
*   [x] `ModelCheckpoint`, `LearningRateMonitor`, `TensorBoardLogger` などの `pytorch_lightning.callbacks` の使用を削除する。

### 7.2.2. 手動での学習ループと評価ループの実装

*   [x] `src/exp/run_teacher.py` 内に、以下の要素を含む標準的なPyTorchの学習ループを実装する。
    *   [x] エポックごとのループ。
    *   [x] `dm.train_dataloader()` からバッチを取得するループ。
    *   [x] バッチデータを適切なデバイス (`cuda` または `cpu`) に移動する処理。
    *   [x] オプティマイザの勾配をゼロにする (`optimizer.zero_grad()`)。
    *   [x] モデルのフォワードパス (`ilora_model_instance(batch)`)。
    *   [x] 損失の計算 (`loss_fn(logits, next_item)`)。
    *   [x] バックワードパス (`loss.backward()`)。
    *   [x] オプティマイザのステップ (`optimizer.step()`)。
    *   [x] 学習損失のログ出力。
*   [x] 各エポックの終わりに、以下の要素を含む手動での評価ループを実装する。
    *   [x] `with torch.no_grad():` コンテキストの使用。
    *   [x] `dm.val_dataloader()` からバッチを取得するループ。
    *   [x] バッチデータを適切なデバイスに移動する処理。
    *   [x] モデルのフォワードパスと損失計算。
    *   [x] `src/core/metrics.py` を使用した評価メトリクス (`recall@k` など) の計算。
    *   [x] 検証損失と評価メリクスのログ出力。

### 7.2.3. モデルの保存とロードの調整

*   [x] 最も良い検証性能を示したモデルの `state_dict` を保存するロジックを実装する。
    *   [x] `best_val_loss` を追跡し、更新された場合に `torch.save(ilora_model_instance.state_dict(), best_model_path)` を実行する。
*   [x] 評価フェーズで、保存されたベストモデルをロードするロジックを実装する。
    *   [x] `ilora_model_instance.load_state_dict(torch.load(best_model_path))` を使用する。

### 7.2.4. その他の調整

*   [x] `src/exp/run_teacher.py` 内で、`iLoRATrainer` の代わりに `ilora_model_instance` を直接使用するようにコードを調整する。
*   [x] `src/exp/run_teacher.py` の評価部分で、`trainer.test(loaded_model, dm)` の代わりに手動でテストループを実装し、`src/student/evaluator.py` の `SASRecEvaluator` を利用して評価メトリクスを計算する。
*   [x] `src/exp/run_teacher.py` の教師出力生成部分で、`loaded_model.model` の代わりに `ilora_model_instance` を直接使用するように調整する。

### 7.2.5. テストの修正と実行

*   [x] `src/exp/run_teacher.py` の変更に伴い、`tests/teacher/test_trainer_ilora.py` を修正する。
    *   [x] `iLoRATrainer` のテストを、`run_teacher.py` の新しい手動ループのロジックを検証するように変更するか、または不要なテストを削除する。 (不要なため削除)
*   [x] `pytest` を実行し、すべてのテストがパスすることを確認する。 (エラー修正中)
    *   [x] `tests/distill/test_trainer_distill.py`, `tests/student/test_datamodule.py`, `tests/student/test_models.py` の `FileNotFoundError` を修正。
    *   [x] `tests/teacher/test_ilora_model.py` の `AssertionError` を修正。
*   [x] `pytest` を再実行し、すべてのテストがパスすることを確認する。

### 7.2.6. 動作確認

*   [ ] 修正後、`docker exec -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" ilora-dev-container bash -c "PYTHONPATH=/workspace poetry run python src/exp/run_teacher.py"` を実行し、エラーなく学習が完了することを確認する。
    *   **現状**: `rec_model.to(device)` または `dummy_model.to(device)` の実行時に `RuntimeError: CUDA driver error: operation not supported` が発生し、学習が完了しません。これはPythonコードの問題ではなく、Dockerコンテナ環境におけるCUDAドライバまたはGPUリソースとの連携に関する根本的な問題である可能性が高いです。`torch.cuda.is_available()` は `True` を返し、PyTorchとCUDAのバージョンは互換性があります。
*   [ ] 学習ログ、保存されたモデル、評価結果が期待通りであることを確認する。

## 7.3. 作業計画 (コンテナ再ビルドによる環境問題の切り分け)

**背景:**
`7.2.6` の調査により、`CUDA driver error: operation not supported` エラーはPythonコードではなく、コンテナ環境とホストGPUの連携に起因する可能性が高いことが判明しました。ホストのNVIDIAドライバとコンテナ内のCUDAバージョンの非互換性が疑われるため、より安定した動作実績のある古いCUDAバージョン（11.8）をベースにしたコンテナを再ビルドし、問題が解決するかを検証します。

### 7.3.1. Dockerfileのベースイメージ変更

*   [x] `Dockerfile` の `FROM` 行を `pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel` から `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel` に変更する。

### 7.3.2. 既存コンテナのクリーンアップ

*   [x] 以下のコマンドを実行し、現在稼働中のコンテナを停止・削除する。
    ```bash
    docker stop ilora-dev-container
    docker rm ilora-dev-container
    ```

### 7.3.3. Dockerイメージの再ビルド

*   [x] 以下のコマンドを実行し、変更後の `Dockerfile` から新しいイメージをビルドする。
    ```bash
    docker build -t ilora-dllm2rec:latest .
    ```

### 7.3.4. コンテナの再起動と動作確認

*   [x] 新しいイメージでコンテナを起動し、`poetry install` を実行する。
*   [ ] `run_teacher.py` を実行し、CUDAエラーが解消されるか確認する。
    ```bash
    docker exec -e PYTORCH_ALLOC_CONF="expandable_segments:True" ilora-dev-container bash -c "PYTHONPATH=/workspace poetry run python /workspace/src/exp/run_teacher.py"
    ```

## 7.4. 作業計画 (OOMエラーの調査と解決)

**背景:**
コンテナ環境をCUDA 11.8ベースに切り替えたことで `CUDA driver error` は解消されたが、新たに `torch.OutOfMemoryError` が発生した。`facebook/opt-125m` という比較的小さなモデルを使用しているにも関わらずメモリ不足が発生しており、単純なモデルサイズ以外の要因を調査する必要がある。

### 7.4.1. バッチサイズの削減による効果測定

*   [x] `conf/train/teacher.yaml` の `batch_size` を `8` に設定して `run_teacher.py` を実行し、OOMが解消されるか確認する。
    *   もし解消された場合、学習が正常に完了するかを見届ける。
    *   解消されない場合、さらにバッチサイズを `1` にして試行する。
*   [x] **動作確認のためのデータ数とエポック数の削減 (一時的)**
    *   [x] `conf/dataset/movielens.yaml` の `limit_data_rows` を `1000` に変更する。
    *   [x] `conf/train/teacher.yaml` の `max_epochs` を `1` に変更する。
    *   [x] 上記設定で `run_teacher.py` を実行し、OOMが解消され、短時間で動作が完了することを確認する。
    *   [x] 動作確認後、`limit_data_rows` と `max_epochs` を元の値に戻す。

### 7.4.2. モデル内部のテンソルサイズ調査 (OOMが解消されない場合)

*   [ ] `src/teacher/ilora_model.py` の `forward` メソッド内に、デバッグコードを追加する。
    *   `inputs_embeds` の `shape` と `dtype` をログ出力し、LLMへの入力テンソルが想定外の大きさになっていないか確認する。
    *   `attention_mask` の `shape` と `dtype` をログ出力する。
*   [ ] `nvidia-smi` コマンドをコンテナ内で定期的に実行し、GPUメモリ使用量の推移を詳細に監視する。

### 7.4.3. 勾配計算の最適化

*   [ ] `torch.no_grad()` の適用範囲を確認する。特に、推論や評価のコードブロックで不要な勾配計算が行われていないか確認する。
*   [ ] `torch.cuda.empty_cache()` を学習ループの適切な場所に追加し、不要なキャッシュがメモリを圧迫していないか確認する。

### 7.4.4. 最終的な動作確認

*   [ ] OOMエラーを解消し、`run_teacher.py` がエラーなく最後まで実行できることを確認する。
*   [ ] 学習ログ、保存されたモデル、評価結果が期待通りであることを確認する。
