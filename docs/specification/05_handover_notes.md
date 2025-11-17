# 引き継ぎノート (Handover Notes)

このドキュメントは、後任のエージェントが本プロジェクトの開発をスムーズに引き継ぐためのものです。

---

## 1. 現状のサマリー

### 1.1. プロジェクト目標

本プロジェクトの最終目標は、**LLM (大規模言語モデル) を教師モデルとして活用し、シーケンシャル推薦モデル (生徒モデル) へ知識蒸留を行うための研究開発基盤を構築すること**です。
`iLoRA` と `DLLM2Rec` の論文で提案されているアーキテクチャを参考に、独自の解釈で実装を進めています。

### 1.2. 実装状況

現在、プロジェクトの基本的な骨格が完成し、各コンポーネントが連携して動作するためのスケルトン実装と、それらを検証する単体テストが完了しています。

- **全体構成**: Hydraによる設定管理、Poetryによる依存関係管理、Dockerによる実行環境が整備されています。
- **ソースコード**: `src/` 配下に、`core`, `student`, `teacher`, `distill`, `exp` の各モジュールが実装済みです。
- **テスト**: `tests/` 配下に各モジュールの単体テストが実装されており、**すべてのテストが成功 (passed) する状態**です。
- **ドキュメント**: `docs/specification` 配下に、概要、開発ノート、テストケース、実行ガイドが整備されています。

**注意**: 教師モデルである `iLoRAModel` は、詳細に実装されています。

---

## 2. プロジェクト構造と規約

### 2.1. 主要ディレクトリ

- `conf/`: Hydraの設定ファイル群。
- `src/`: Pythonのソースコード。
  - `core/`: プロジェクト共通のユーティリティ。
  - `student/`: 生徒モデル (`SASRec`) 関連。
  - `teacher/`: 教師モデル (`iLoRAModel`) 関連。
  - `distill/`: 知識蒸留のロジック関連。
  - `exp/`: 実験実行のエントリポイント。
- `tests/`: `pytest`によるテストコード。
- `docs/`: 設計書や仕様書。
- `cmd/`: 各種実行スクリプト。
- `ref_repositories/`: 参考にした論文のコードリポジトリ（直接のインポートは禁止）。

### 2.2. 重要な規約

- **設計書の遵守**: `docs/implement.md` は本プロジェクトの唯一の設計書です。**このファイルの編集は禁止**されています。
- **開発記録**: 機能追加や大きな変更を行った際は、`docs/specification/02_development_notes_ja.md` に思考プロセスや実装内容を記録してください。
- **参考コードの扱い**: `ref_repositories/` 内のコードはあくまで参考資料です。ロジックを理解した上で、本プロジェクトの設計に合わせて再実装してください。直接のコピー＆ペーストやインポートは禁止です。

---

## 3. 開発・実行環境

### 3.1. Dockerコンテナ

開発は `ilora-dev-container` という名前のDockerコンテナ内で完結します。

- **コンテナの起動**:
  ```bash
  docker run -d --name ilora-dev-container -v "$(pwd)":/workspace -w /workspace --gpus all -it ilora-dllm2rec:latest
  ```
- **コンテナ内でのコマンド実行**:
  ```bash
  docker exec -it ilora-dev-container bash
  ```

### 3.2. テストの実行

テストは `pytest` を使用します。コンテナ内で以下のコマンドを実行してください。

```bash
# 個別テストの実行
poetry run pytest tests/teacher/test_ilora_model.py

# 全テストの実行
poetry run pytest
```
**重要**: Pythonモジュールを実行する際は、`PYTHONPATH` が正しく設定されている必要があります。`docker exec` でコマンドを実行する場合、`PYTHONPATH=/workspace` を明示的に指定し、`-m` オプションでモジュールとして実行することを推奨します。

**例: `src/teacher/ilora_model.py` を実行する場合**
```bash
docker exec ilora-dev-container bash -c "PYTHONPATH=/workspace poetry run python -m src.teacher.ilora_model"
```

すべてのテストがパスすることを確認しながら開発を進めてください。

---

## 4. 残されたタスクと次のステップ

### 4.1. 今後のタスクリスト

1.  **教師モデル (`iLoRAModel`) の詳細実装**:
    -   [x] **プロンプトエンジニアリング**: アイテム履歴を自然言語プロンプトに変換するロジックの実装。
    -   [x] **動的なLoRAアダプターの結合**: `peft`ライブラリを活用した、より効率的なフォワードパスの実装。
    -   [x] **出力のマッピング**: LLMの出力ロジットから推薦アイテムのスコアを計算するロジックの洗練。

2.  **データ関連処理の強化**:
    -   [x] **データブリッジ (`src/distill/data_bridge.py`) の実装**: 教師モデルと生徒モデル間のデータ形式の違いを吸収する層の実装。
    -   [x] **選択的蒸留 (`src/distill/selection_policy.py`) の高度化**: より効果的な蒸留サンプルを選択する高度なポリシーの実装。

3.  **実験と評価**:
    -   [x] **チェックポイント管理**: 学習済みモデルの保存・ロード機能の具体化。
    -   [x] **実験の実行と分析**: `cmd/` スクリプトを用いた本格的な実験の実施。

### 4.2. 次にすべきこと

上記のタスクリストに基づき、**次に着手すべき最優先タスクは「DLLM2Rec DRO損失の再現」**です。

具体的には、`docs/specification/06_difference_from_asis.md` に記載されているDLLM2Recのロジック差分を参考に、既存実装を修正してください。特に、DLLM2Recのランキング蒸留における複雑な重み付けロジックや、埋め込み蒸留の適用方法については、先行研究の意図を深く理解し、再現性を高めることを優先してください。

*   **DLLM2Rec DRO損失の再現**:
    *   `ref_repositories/DLLM2Rec/main.py` に実装されているDRO損失を `src/distill/kd_losses.py` に追加し、`src/distill/trainer_distill.py` で使用するように修正してください。
*   **設定ファイルの更新**:
    *   DLLM2Recのロジック再現に必要な新しいハイパーパラメータ（例: `ed_weight`, `lam`, `gamma_position` など）を `conf/distill/dllm2rec.yaml` に追加し、`src/distill/trainer_distill.py` のコンストラクタで受け取るように修正してください。
*   **テストの追加と修正**:
    *   DLLM2Recロジックの変更に伴い、`tests/distill/test_kd_losses.py` および `tests/distill/test_trainer_distill.py` に新しいテストケースを追加し、既存のテストケースを修正して、変更が正しく機能することを確認してください。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。

---

## 5. エージェントによる引き継ぎノート (2025-11-17)

### 5.1. 実施した作業の概要

*   **データセット行数制限機能の導入**:
    *   `conf/dataset/movielens.yaml` に `limit_data_rows` パラメータを追加し、データローダーが読み込むデータ行数を制限できるようにしました。
    *   `src/exp/run_teacher.py` および `src/exp/run_distill.py` がこのパラメータを `SASRecDataModule` に正しく渡すように修正しました。
    *   これにより、エポックあたりのステップ数を制御し、デバッグ時の実行時間を短縮できるようになりました。
    *   関連ドキュメント (`docs/specification/02_development_notes_ja.md`, `docs/specification/04_execution_guide.md`) を更新しました。
*   **教師モデル (`iLoRAModel`) のメモリ最適化**:
    *   `src/teacher/ilora_model.py` において、`item_embeddings` 層が `llm.config.vocab_size` ではなく `hidden_size` を使用するように変更し、さらにLLMの入力次元に合わせるためのプロジェクション層 (`item_embedding_projection`) を追加しました。これにより、`CUDA out of memory` エラーを解消しました。
    *   `src/teacher/factory.py` を修正し、`iLoRAModel` のコンストラクタに `llm` と `tokenizer` オブジェクトを渡すように変更しました。
*   **蒸留選択ポリシーの修正**:
    *   `src/distill/selection_policy.py` に `AllSamplesPolicy` クラスを追加しました。これは、`src/exp/run_distill.py` での `ImportError` を解消するためです。
*   **メトリクス計算の修正**:
    *   `src/distill/trainer_distill.py` において、`calculate_metrics` 関数に渡す `logits` と `next_item` の形式を `List[List[int]]` に変換するように修正しました。これにより、`RuntimeError: Boolean value of Tensor with more than one value is ambiguous` エラーを解消しました。
*   **教師モデルのチェックポイントパスの更新**:
    *   `run_teacher.py` の実行後、生成された教師モデルのチェックポイントパス (`/workspace/result/result_20251117_011905/checkpoints/best_teacher_model.ckpt`) を `conf/distill/dllm2rec.yaml` の `teacher_checkpoint_path` に設定しました。
*   **コンテナ起動・依存関係インストールスクリプトの作成**:
    *   `cmd/start_container_and_install.sh` を作成し、コンテナの起動と `poetry install` の実行を自動化しました。これにより、`docs/specification/04_execution_guide.md` に記載されている「コンテナを起動してからコンテナ内で実行するレギュレーション」に対応しました。
*   **`ModuleNotFoundError` の解決**:
    *   `cmd/run_distill.sh` に `PYTHONPATH=/workspace` を追加することで、`src/teacher/ilora_model.py` 内での `src.teacher.mlp_projector` のインポートに関する `ModuleNotFoundError` を解決しました。
*   **蒸留実験の正常完了**:
    *   上記の `ModuleNotFoundError` 解決後、`cmd/run_distill.sh` を実行し、蒸留実験が正常に完了することを確認しました。
*   **`iLoRAModel` の `get_teacher_outputs` の `embeddings` の確認**:
    *   `src/teacher/ilora_model.py` の `get_teacher_outputs` メソッドにおける `embeddings` の返却方法を確認しました。現在の実装 (`combined_hidden_states`) は、各LoRAエキスパートの最終隠れ状態をゲーティングネットワークの重みで結合したものであり、LLMの最終隠れ状態を適切に集約していると判断しました。このタスクは完了とします。

### 5.2. 現在の課題と次のエージェントへの依頼事項

### 5.2. 現在の課題と次のエージェントへの依頼事項

*   **DLLM2Rec 埋め込み蒸留ロジックの再現**:
    *   `ref_repositories/DLLM2Rec/main.py` では、LLMからの埋め込みを生徒モデルの入力埋め込みに直接加算することで埋め込み蒸留を行っています。このロジックは `src/student/models.py` の `SASRec` モデルに移植され、`src/distill/trainer_distill.py` でLLM埋め込みを `SASRec` モデルに渡すことで実装済みです。
*   **DLLM2Rec DRO損失の再現**:
    *   `ref_repositories/DLLM2Rec/main.py` に実装されているDRO損失を `src/distill/kd_losses.py` に追加し、`src/distill/trainer_distill.py` で使用するように修正してください。
*   **設定ファイルの更新**:
    *   DLLM2Recのロジック再現に必要な新しいハイパーパラメータ（例: `ed_weight`, `lam`, `gamma_position` など）を `conf/distill/dllm2rec.yaml` に追加し、`src/distill/trainer_distill.py` のコンストラクタで受け取るように修正してください。
*   **テストの追加と修正**:
    *   DLLM2Recロジックの変更に伴い、`tests/distill/test_kd_losses.py` および `tests/distill/test_trainer_distill.py` に新しいテストケースを追加し、既存のテストケースを修正して、変更が正しく機能することを確認してください。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。

---

### 5.5. エージェントによる引き継ぎノート (2025-11-17, 2回目)

#### 5.5.1. 実施した作業の概要

*   **iLoRAロジックのリファクタリング**:
    *   参照リポジトリを基に、`src/teacher/gating.py`と`src/teacher/moe_lora_model.py`を作成し、iLoRAの主要な構成要素を移植しました。
    *   `src/teacher/ilora_model.py`を修正し、新しく作成した`MoeLoraModel`を使用するように変更しました。
*   **テストスイートの修正**:
    *   上記のリファクタリングに伴い、`tests`ディレクトリ配下の複数のテストファイル（`test_trainer_distill.py`, `test_ilora_model.py`, `test_trainer_ilora.py`など）を修正し、`NameError`, `TypeError`, `AttributeError`などのエラーを解決しました。
    *   `SASRecDataModule`が`tokenizer`を必要とするようになったため、関連するテストフィクスチャを更新しました。
*   **`RuntimeError`の解決と`nn.Embedding`の勾配追跡問題**:
    *   `DistillationTrainer`の`training_step`で発生していた`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`を解決しました。
    *   `nn.Embedding`の初期化における`IndexError`を修正し、`nn.Embedding(num_items + 2, hidden_size, padding_idx=padding_item_id)`に戻しました。
    *   `item_embeddings.requires_grad`が`False`となる問題に対し、`SASRec._get_last_item_representation`内で`item_embeddings.requires_grad_(True)`を追加して勾配追跡を強制しました。
    *   `item_embeddings += ...`というインプレース操作が原因で発生していた`RuntimeError`を、`item_embeddings = item_embeddings + ...`というアウトオブプレース操作に変更することで解決しました。
*   **DLLM2Rec 埋め込み蒸留ロジックの再現**:
    *   `ref_repositories/DLLM2Rec/main.py` で示されている、LLMからの埋め込みを生徒モデルの入力埋め込みに直接加算する埋め込み蒸留ロジックを `src/student/models.py` の `SASRec` モデルに移植し、`src/distill/trainer_distill.py` でLLM埋め込みを `SASRec` モデルに渡すように修正しました。

#### 5.5.2. 現在の課題と次のエージェントへの依頼事項

iLoRAロジックのリファクタリングと関連するテストの修正は完了し、教師モデルの学習が正常に実行されることを確認しました。
次のエージェントは、DLLM2Recロジックの再現性向上に着手してください。

*   **DLLM2Rec DRO損失の再現**:
    *   `ref_repositories/DLLM2Rec/main.py` に実装されているDRO損失を `src/distill/kd_losses.py` に追加し、`src/distill/trainer_distill.py` で使用するように修正してください。
*   **設定ファイルの更新**:
    *   DLLM2Recのロジック再現に必要な新しいハイパーパラメータ（例: `ed_weight`, `lam`, `gamma_position` など）を `conf/distill/dllm2rec.yaml` に追加し、`src/distill/trainer_distill.py` のコンストラクタで受け取るように修正してください。
*   **テストの追加と修正**:
    *   DLLM2Recロジックの変更に伴い、`tests/distill/test_kd_losses.py` および `tests/distill/test_trainer_distill.py` に新しいテストケースを追加し、既存のテストケースを修正して、変更が正しく機能することを確認してください。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。