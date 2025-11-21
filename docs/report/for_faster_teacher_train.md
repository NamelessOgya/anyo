# 教師モデルの学習高速化に関する実験結果

## 1. 概要

教師モデル (iLoRA) の学習および推論 (教師出力生成) プロセスを高速化するために実施された実験の結果をまとめる。特に、半精度学習 (BF16) の導入効果を定量的に評価した。

## 2. 評価環境

*   **GPUモデル**: NVIDIA A100-SXM4-80GB
*   **GPUメモリ**: 80GB (現在使用量: 0MiB)
*   **GPU使用率**: 0%
*   **Driver Version**: 550.54.15
*   **CUDA Version**: 12.4

## 3. ベンチマーク結果

`limit_data_rows: 10000` (`max_epochs: 1` または `2`) の設定で実施。

### 3.1. 生徒ベースラインモデル - Initial Benchmark

*   **実行日時**: 2025-11-20 09:00:57
*   **学習時間**: 5分6.184秒 (real)
*   **最終チェックポイント**: `/content/drive/MyDrive/rec/anyo/result/student_baseline_20251120_090057/checkpoints/student-baseline-epoch=01-val_recall@10=0.0061.ckpt`
*   **テスト結果**:
    *   `test_hit_ratio@10`: 0.00546
    *   `test_loss`: 15.3876
    *   `test_ndcg@10`: 0.00215
    *   `test_recall@10`: 0.00546

### 3.2. 教師モデル (iLoRA) - FP32学習 - Precise Timing

*   **実行日時**: 2025-11-20 10:06:37
*   **総実行時間 (real)**: 6分23.344秒
*   **学習 (`trainer.fit`) 時間**: 139.44秒 (2分19.44秒)
*   **エポック学習時間 (Epoch 0)**: 41.44秒
*   **テストセット評価時間**: 106.33秒 (1分46.33秒)
*   **教師出力生成時間**: 21.74秒
*   **ピークGPUメモリ使用量**: 5.06 GB (Epoch 0 終了時)
*   **最終チェックポイント**: `/content/drive/MyDrive/rec/anyo/result/teacher_20251120_100637/checkpoints/teacher-epoch=00-val_loss=8.1937.ckpt`
*   **教師出力生成ディレクトリ**: `/content/drive/MyDrive/rec/anyo/result/teacher_20251120_100637/teacher_outputs_batches`
*   **テスト結果**:
    *   `test_hit_ratio@10`: 0.009768
    *   `test_loss`: 8.200447
    *   `test_ndcg@10`: 0.004178
    *   `test_recall@10`: 0.009768

### 3.3. 教師モデル (iLoRA) - BF16学習 - Precise Timing

*   **実行日時**: 2025-11-20 09:57:50
*   **総実行時間 (real)**: 6分22.716秒
*   **学習 (`trainer.fit`) 時間**: 139.70秒 (2分19.70秒)
*   **エポック学習時間 (Epoch 0)**: 41.21秒
*   **テストセット評価時間**: 105.66秒 (1分45.66秒)
*   **教師出力生成時間**: 21.94秒
*   **ピークGPUメモリ使用量**: 5.06 GB (Epoch 0 終了時)
*   **FP32との比較**:
    *   **学習時間**: 139.44秒 (FP32) vs 139.70秒 (BF16) - 顕著な差なし
    *   **テストセット評価時間**: 106.33秒 (FP32) vs 105.66秒 (BF16) - 顕著な差なし
    *   **教師出力生成時間**: 21.74秒 (FP32) vs 21.94秒 (BF16) - 顕著な差なし
    *   **ピークGPUメモリ使用量**: 5.06 GB (FP32) vs 5.06 GB (BF16) - 変化なし
    *   **備考**: `torch.cuda.max_memory_allocated()`による計測では、FP32とBF16でピークメモリ使用量および実行時間に顕著な変化が見られませんでした。これは一般的な挙動とは異なり、実装または計測方法に誤りがある可能性があります。後続の最適化が完了した後、改めて調査・検証が必要です。
*   **最終チェックポイント**: `/content/drive/MyDrive/rec/anyo/result/teacher_20251120_095750/checkpoints/teacher-epoch=00-val_loss=8.1946.ckpt`
*   **教師出力生成ディレクトリ**: `/content/drive/MyDrive/rec/anyo/result/teacher_20251120_095750/teacher_outputs_batches`
*   **テスト結果**:
    *   `test_hit_ratio@10`: 0.009437
    *   `test_loss`: 8.20170
    *   `test_ndcg@10`: 0.00408
    *   `test_recall@10`: 0.009437

### 3.4. 教師モデル (iLoRA) - KVキャッシュ有効化 (explicit `use_cache=True`)

*   **実行日時**: 2025-11-20 10:50:06
*   **教師出力生成時間**: 22.09秒
*   **FP32学習 (`use_cache`デフォルト) との比較**:
    *   **教師出力生成時間**: 21.74秒 (デフォルト) vs 22.09秒 (explicit `use_cache=True`) - 顕著な差なし (わずかに増加)
    *   **備考**: 明示的に `use_cache=True` を設定しても、この小規模なテスト (`limit_data_rows: 10000`) では教師出力生成時間に顕著な改善は見られませんでした。これは、デフォルトで既に有効になっていたか、オーバーヘッドがパフォーマンス向上を上回った可能性があります。

### 3.5. 教師モデル (iLoRA) - 推論バッチサイズ最適化 (`inference_batch_size=256`)

*   **実行日時**: 2025-11-20 11:04:17
*   **教師出力生成時間**: 19.76秒
*   **比較**:
    *   **教師出力生成時間 (デフォルトバッチサイズ=64)**: 22.09秒
    *   **教師出力生成時間 (inference_batch_size=256)**: 19.76秒 (約10%削減)
    *   **備考**: 推論時のバッチサイズを64から256に増やすことで、教師出力生成時間が約10%削減されました (22.09秒 -> 19.76秒)。これは、GPUをより効率的に利用できたことを示します。

### 3.6. 教師モデル (iLoRA) - `torch.compile` の導入検討

*   **実行日時**: 2025-11-20 11:12:32 (実行開始)
*   **備考**: `torch.compile`の導入を試みましたが、実行時間が30分を超過し、即座の改善が見られなかったため、本タスクはこれ以上の深掘りを行わずに完了と判断されました。

### 3.7. 教師モデル (iLoRA) - データローダーの `num_workers` の最適化 (`num_workers=4`)

*   **実行日時**: 2025-11-20 11:29:32
*   **学習 (`trainer.fit`) 時間**: 142.15秒 (2分22.15秒)
*   **教師出力生成時間**: 19.81秒
*   **比較 (`num_workers=0` の場合)**:
    *   **学習時間**: 139.44秒 (FP32 baseline) vs 142.15秒 (`num_workers=4`) - わずかに増加
    *   **教師出力生成時間**: 19.76秒 (`num_workers=0`) vs 19.81秒 (`num_workers=4`) - わずかに増加
    *   **備考**: `num_workers`を4に増やしましたが、小規模データセットではパフォーマンスの改善は見られず、むしろ学習時間がわずかに増加しました。これは、データセットが小さいためデータローディングがボトルネックになっていないか、あるいはマルチプロセスによるオーバーヘッドが原因である可能性があります。また、`huggingface/tokenizers`の並列処理に関する警告も発生しました。この設定は現状では効果がないと判断し、`num_workers=0`に戻します。

### 3.8. 教師モデル (iLoRA) - QLoRAの導入検討 (キャンセル済み)

*   **試行期間**: 2025-11-20 (複数回試行)
*   **概要**: 教師モデル (`iLoRAModel`) におけるQLoRA (4-bit量子化) の導入を試みました。これは、LLMのメモリ使用量を削減し、学習・推論を高速化することを目的としていました。`BitsAndBytesConfig`と`peft`ライブラリを組み合わせて実装を進めました。
*   **遭遇した課題**:
    1.  **`RuntimeError: self and mat2 must have the same dtype, but got Half and Byte`**: `bitsandbytes`がラップする線形層への入力のdtypeが`torch.float16` (`Half`) であるのに対し、内部の重みが`torch.uint8` (`Byte`) であり、計算dtypeとして期待される`torch.bfloat16`と合致しないために発生しました。`bfloat16`の強制適用、`BitsAndBytesConfig`の明示的な使用、`llm.bfloat16()`によるモデル全体の変換、`moe_lora_model.py`内の入力の明示的なキャストなど、様々なアプローチを試みましたが、問題の根本解決には至りませんでした。
    2.  **`RuntimeError: Error(s) in loading state_dict for iLoRATrainer: Unexpected key(s) in state_dict: ...`**: 上記のdtypeの問題を解決するために`peft.prepare_model_for_kbit_training`を導入したところ、トレーニング自体は進行しましたが、モデルチェックポイントのロード時に新たなエラーが発生しました。これは、`peft`がラップしたLLMの`state_dict`キー（例: `llm.base_model.model.model.decoder.layers.0.self_attn.k_proj.weight.absmax`など、`bitsandbytes`の量子化情報や`peft`のアダプター層に関するキー）が、`pytorch_lightning.LightningModule`の標準的な`load_state_dict`の期待する構造と一致しないために生じました。手動で`state_dict`をフィルタリングしてロードする試みも行いましたが、複雑なモデル構造と`peft`の内部的な処理により、完全に解決することができませんでした。
*   **結論**: カスタムのMoE LoRA実装とPEFTのQLoRA統合が期待通りに動作せず、既存のLoRA推論機能との複雑な相互作用を引き起こしていることが判明しました。これらの問題は根深く、単純な修正では対応が困難であるため、本タスクは一旦キャンセルとし、将来的に詳細な調査とより抜本的なリファクタリングが必要と判断しました。
*   **メモリ使用量/速度**: 上記の問題が解決されていないため、QLoRA導入によるメモリ使用量削減および速度向上に関するベンチマーク結果は得られませんでした。


