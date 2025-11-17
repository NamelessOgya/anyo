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
- **ソースコード**: `src/` 配下に、`core`, `student`, `teacher`, `distill`, `exp` の各モジュールがスケルトンとして実装済みです。
- **テスト**: `tests/` 配下に各モジュールの単体テストが実装されており、**すべてのテストが成功 (passed) する状態**です。
- **ドキュメント**: `docs/specification` 配下に、概要、開発ノート、テストケース、実行ガイドが整備されています。

**注意**: 教師モデルである `iLoRAModel` は、まだ多くの部分が概念実証のためのダミー実装（ランダムな値を返すなど）となっています。

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
    -   [ ] **チェックポイント管理**: 学習済みモデルの保存・ロード機能の具体化。
    -   [ ] **実験の実行と分析**: `cmd/` スクリプトを用いた本格的な実験の実施。

### 4.2. 次にすべきこと

上記のタスクリストに基づき、**次に着手すべき最優先タスクは「3. 実験と評価」**です。

具体的には、以下のタスクから着手してください。

-   **チェックポイント管理**: 学習済みモデルの保存・ロード機能の具体化。
-   **実験の実行と分析**: `cmd/` スクリプトを用いた本格的な実験の実施。

これらの実装を進めることで、プロジェクトの実験パイプラインが完成に近づきます。

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

### 5.2. 現在の課題と次のエージェントへの依頼事項

*   **`ModuleNotFoundError: No module named 'src.teacher.mlp_projector'` の継続**:
    *   `src/teacher/ilora_model.py` 内での `MLPProjector` のインポートに関して、`from .mlp_projector import MLPProjector` および `from src.teacher.mlp_projector import MLPProjector` の両方を試しましたが、`python -m src.exp.run_distill` 実行時に `ModuleNotFoundError` が解消されていません。
    *   この問題は、Pythonのモジュール解決メカニズムと `PYTHONPATH` の設定、および `python -m` での実行方法の組み合わせに起因している可能性があります。
    *   **次のエージェントへの依頼**: この `ModuleNotFoundError` の根本原因を特定し、解決してください。`src/teacher/mlp_projector.py` が正しくインポートされるように、インポートパスまたはプロジェクトの実行方法を調整する必要があります。
*   **蒸留実験の完了**:
    *   上記の `ModuleNotFoundError` が解決され次第、蒸留実験 (`src/exp/run_distill.py`) を正常に完了させてください。
    *   `limit_data_rows` は現在 `160` に設定されており、`batch_size: 32` の場合、1エポックあたり約5ステップで実行されるはずです。
*   **`iLoRAModel` の `get_teacher_outputs` の改善**:
    *   `src/teacher/ilora_model.py` の `get_teacher_outputs` メソッドでは、現在 `embeddings` として最後のアイテムの埋め込みを一時的に返しています。LLMの最終隠れ状態を適切に返すように修正する必要があります（コード内の `TODO` コメントを参照）。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。

### 5.3. 最終的なファイルの状態

*   `conf/dataset/movielens.yaml`: `limit_data_rows: 160`
*   `conf/distill/dllm2rec.yaml`: `teacher_checkpoint_path` が設定済み
*   `src/exp/run_teacher.py`: `limit_data_rows` を `SASRecDataModule` に渡すように修正済み
*   `src/exp/run_distill.py`: `limit_data_rows` を `SASRecDataModule` に渡すように修正済み
*   `src/distill/selection_policy.py`: `AllSamplesPolicy` を追加済み
*   `src/distill/trainer_distill.py`: `calculate_metrics` の引数変換を修正済み
*   `src/teacher/ilora_model.py`: `item_embeddings` のメモリ最適化、`output_layer` の次元修正、`MLPProjector` のインポート問題が残存
*   `src/teacher/factory.py`: `llm`, `tokenizer`, `MLPProjector` を `iLoRAModel` に渡すように修正済み
*   `docs/specification/02_development_notes_ja.md`: `limit_data_rows` の説明を追加済み
*   `docs/specification/04_execution_guide.md`: `limit_data_rows` とメモリ問題に関する注意を追加済み

この引き継ぎノートが、次のエージェントの開発をスムーズに進める一助となることを願っています。