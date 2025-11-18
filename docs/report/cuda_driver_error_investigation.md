# 教師モデル学習時の `CUDA driver error` に関する調査報告書

## 1. 問題の概要

教師モデルの学習スクリプト (`src/exp/run_teacher.py`) の実行中に、`RuntimeError: CUDA driver error: operation not supported` という低レベルなCUDAドライバーエラーが継続して発生しました。このエラーは、`src/teacher/factory.py` 内で、事前学習済みの `SASRec` モデル (`rec_model`) をCUDAデバイスに移動しようとする `rec_model.to(device)` の呼び出し時に発生します。

## 2. 調査の経緯と実施した対策

このエラーを解決するため、以下の多岐にわたる調査と対策を実施しました。

### 2.1. メモリ使用量の調査 (OOMエラーの切り分け)

当初はOOM (Out of Memory) エラーが頻発していたため、メモリ関連の問題を中心に調査しました。

*   **最小限のPyTorchスクリプトの作成**:
    *   PyTorch Lightningのトレーナーを使用せず、`iLoRAModel` の単一のフォワードパスとバックワードパスを実行する `src/exp/debug_teacher_oom.py` を作成しました。
    *   **結果**: `batch_size=32` でもOOMエラーは発生せず、メモリ使用量もGPU容量内に収まることを確認しました。これにより、問題がモデルの単一ステップのメモリフットプリントではなく、PyTorch Lightningのトレーナーや学習ループ全体に関連する可能性が示唆されました。

*   **アプリケーションレベルでのメモリ最適化**:
    *   **勾配チェックポインティング**: `iLoRAModel` で有効化しましたが、`RuntimeError: Trying to backward through the graph a second time` という別のエラーが発生したため、無効化しました。
    *   **バッチサイズとワーカー数の削減**: `conf/train/teacher.yaml` の `batch_size` を `32` に、`num_workers` を `0` に設定しました。
    *   **メモリ断片化対策**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 環境変数を設定しました。

### 2.2. `CUDA driver error` への対策

OOMエラーが解消された後も `CUDA driver error` が継続したため、デバイスへのモデル配置方法を中心に調査しました。

*   **冗長な `.to(device)` 呼び出しの削除**: `src/exp/run_teacher.py` での冗長な呼び出しを削除しました。
*   **LLM/Tokenizerのロードコンテキストの変更**: `src/exp/run_teacher.py` でロードしていたLLMとTokenizerを、`src/teacher/factory.py` 内でロードするように変更し、デバイス配置のコンテキストを統一しました。
*   **`rec_model` のロード方法の変更**: `torch.load` の `map_location` を `'cpu'` に設定し、`rec_model` のチェックポイントを一度CPUにロードしてからGPUに移動するように変更しました。
*   **`torch.set_float32_matmul_precision` の設定**: PyTorchからの警告に基づき、`'high'` を設定しました。
*   **インデントエラーの修正**: 調査の過程で発生した `IndentationError` を修正しました。

## 3. 結論

上記のすべての対策を講じましたが、`RuntimeError: CUDA driver error: operation not supported` は解消されませんでした。エラーは常に同じ箇所 (`rec_model.to(device)`) で発生し続けています。

最小限のスクリプトでは問題が再現されないことから、コードのロジック自体に根本的な欠陥がある可能性は低いと考えられます。このエラーは、PyTorch Lightningの特定の内部動作と、現在の実行環境（CUDAドライバー、GPUハードウェア、Dockerの相互作用）との間に非互換性が存在することを示唆しています。

このような低レベルな環境問題は、アプリケーションコードの修正のみで解決することは極めて困難です。したがって、この問題は現在の環境における**ブロック要因**であると結論付けられます。

## 4. 今後の推奨事項

*   **実行環境の見直し**:
    *   ホストマシンのNVIDIAドライバーのバージョンと、Dockerコンテナ内のPyTorchが要求するCUDAのバージョンに互換性があるか確認する。
    *   可能であれば、異なるバージョンのNVIDIAドライバー、CUDA Toolkit、PyTorch、またはPyTorch Lightningで実験を試みる。
*   **問題の単純化**:
    *   `iLoRAModel` から `rec_model` の部分を一時的に切り離し、LLM部分のみで学習が実行できるかを確認する。これにより、問題が `rec_model` との相互作用に起因するのかを特定できる可能性があります。
