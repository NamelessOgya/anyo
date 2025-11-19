# 5. エージェント引き継ぎドキュメント

## 1. 現在のプロジェクト状況

### 1.1. 全体目標

教師モデルの学習パイプラインを安定稼働させ、環境問題やメモリ問題、数値不安定性の問題を解決すること。

### 1.2. 現在の進捗

*   **PyTorch Lightning実装への回帰:** コードベースは、以前の安定していたPyTorch Lightningベースの実装に復元済み。
*   **Docker環境の改善:** `CUDA driver error` は、Dockerイメージを `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel` に変更し、コンテナを再ビルドすることで解決済み。
*   **デバイス不一致エラーの修正:** 教師出力生成フェーズで発生していた `RuntimeError: Expected all tensors to be on the same device` は、`run_teacher.py` 内でモデルとバッチデータを明示的に同じデバイスに送るように修正することで解決済み。
*   **数値不安定性 (`NaN` / `inf` loss) の問題:**
    *   学習率の引き下げ (`1e-3` -> `1e-4`)、`precision=32` の強制、勾配クリッピング (`gradient_clip_val=1.0`) の導入を試みたが、`NaN` 問題は解決せず。
    *   `validation_step` および `test_step` でのデバッグプリントにより、`logits` が `NaN` または `inf` になっていることが確認された。
    *   この問題は、学習の早い段階（最初のエポックの検証フェーズ）で発生しており、モデルの出力が発散していることが示唆されている。
    *   **現在の仮説:** LLMからの出力である `last_hidden_state` が訓練中に発散し、それが後続の `item_prediction_head` を経て `logits` の `inf`/`NaN` を引き起こしている可能性が高い。

## 2. 次の作業計画

`docs/specification/07_tmp_plan.md` に詳細な計画が記載されていますが、主要な次のステップは以下の通りです。

### 2.1. 数値不安定性 (`NaN` / `inf` loss) の根本原因調査と解決

*   `validation_step` 内で `last_hidden_state` の統計情報（最小値、最大値、平均値）をプリントし、実際に値が発散しているかを確認する。
*   発散が確認された場合、以下の可能性を調査し、解決策を検討する。
    *   `item_prediction_head` の初期化方法に問題がないか。
    *   事前学習済み `rec_model` が不安定性の原因となっていないか。
    *   モデルのアーキテクチャ自体に数値不安定性を引き起こす要因がないか。

### 2.2. 開発サイクル改善: プログレスバーの導入

*   PyTorch Lightning と `rich` を組み合わせたプログレスバー実装のベストプラクティスを調査し、`docs/research/01_progressbar_best_practice.md` にまとめる。
*   `rich` をプロジェクトの依存関係に追加し、`pytorch_lightning.callbacks.RichProgressBar` を使用して、`run_teacher.py`, `run_student_baseline.py`, `run_distill.py` の各学習スクリプトに適用する。
*   データ数とエポック数を少なく設定した上で、新しいプログレスバーが正しく機能することを確認する。

## 3. 環境情報

*   **OS:** Linux
*   **現在の作業ディレクトリ:** `/home/ubuntu/v2`
*   **Dockerコンテナ名:** `ilora-dev-container`
*   **Dockerイメージ:** `ilora-dllm2rec:latest` (ベースイメージ: `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel`)
*   **Python環境:** `poetry` で管理されており、コンテナ内で `poetry install` を実行する必要がある。

## 4. その他

*   `docs/research/01_progressbar_best_practice.md` は既に作成済みです。
*   `pyproject.toml` に `rich` を追加し、`poetry lock` と `poetry install` を実行済みです。
*   `src/exp/run_teacher.py`, `src/exp/run_student_baseline.py`, `src/exp/run_distill.py` のプログレスバー関連の変更は、ユーザーの指示により一旦元に戻されています。
*   `conf/dataset/movielens.yaml` の `limit_data_rows` は `100` に設定されたままです。
*   `conf/train/teacher.yaml` の `learning_rate` は `1e-4` に設定されたままです。
*   `src/exp/run_teacher.py` の `precision=32` と `gradient_clip_val=1.0` の設定は元に戻されています。
*   `src/teacher/trainer_ilora.py` のデバッグプリントは元に戻されています。

次のエージェントは、上記の状況を理解し、`2.1` の数値不安定性問題の調査から開始してください。
