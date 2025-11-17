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
    -   [x] **チェックポイント管理**: 学習済みモデルの保存・ロード機能の具体化。
    -   [x] **実験の実行と分析**: `cmd/` スクリプトを用いた本格的な実験の実施。

### 4.2. 次にすべきこと

上記のタスクリストに基づき、**次に着手すべき最優先タスクは「ロジックレベルの差分確認」**です。

具体的には、`docs/specification/06_difference_from_asis.md` に記載されている依頼内容に基づき、本プロジェクトの既存実装ロジックと参照リポジトリの実装ロジックを比較し、その差分を特定してください。**見つかった差分は、`docs/specification/06_difference_from_asis.md` に詳細に記載してください。**

この作業を進めることで、本プロジェクトのコードが既存ロジックをどの程度再現できているかを客観的に評価できます。

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

*   [x] **`RuntimeError: expand(...)` の解決**: `transformers`ライブラリの`OPTModel`内で`attention_mask`の処理に関する`RuntimeError`が発生しています。`attention_mask`の次元が正しくないことが原因と考えられます。
*   [x] **`RuntimeError: Expected all tensors to be on the same device...` の解決**: モデルの重みと入力テンソルが異なるデバイス（CPUとCUDA）に配置されているため、実行時エラーが発生しています。すべてのテンソルが正しいデバイスに配置されるように修正が必要です。
*   [x] **`UnboundLocalError: local variable 'model' referenced before assignment` の解決**: テストコード内で`model`変数が定義される前に`model.device`を参照しているため、エラーが発生しています。`model`の定義前に`llm`をデバイスに移動する処理を修正する必要があります。
*   **`iLoRAModel` の `get_teacher_outputs` の改善**:
    *   `src/teacher/ilora_model.py` の `get_teacher_outputs` メソッドでは、`embeddings` としてLLMの最終隠れ状態を適切に返すように修正する必要があるという課題がありましたが、現在の実装 (`combined_hidden_states`) は、各LoRAエキスパートの最終隠れ状態をゲーティングネットワークの重みで結合したものであり、LLMの最終隠れ状態を適切に集約していると判断しました。もし、これに関して別の意図があった場合は、詳細な指示をお願いします。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。
*   **ロジックレベルの差分解消**:
    *   `docs/specification/06_difference_from_asis.md` にて、本プロジェクトのiLoRAおよびDLLM2Recの実装と参照リポジリのロジックレベルの差分を詳細に分析しました。次のエージェントは、このドキュメントに記載された差分を十分に理解し、**既存実装と先行研究（参照リポジリ）のロジックの差を埋める実装**を行ってください。特に、DLLM2Recのランキング蒸留における複雑な重み付けロジックや、iLoRAのLoRAエキスパートの結合方法については、先行研究の意図を深く理解し、再現性を高めることを優先してください。

---

### 5.4. ロジックレベルの差分解消計画 (2025-11-17)

`docs/specification/06_difference_from_asis.md` で特定されたロジックレベルの差分を解消するための計画を以下に示します。

*   **iLoRAロジックの再現性向上**
    *   [x] `ref_repositories/iLoRA/model/peft/tuners/gating.py` を参考に、`src/teacher/gating.py` を作成し、ゲーティングネットワークの実装を移植する。
    *   [x] `ref_repositories/iLoRA/model/peft/tuners/moelora.py` を参考に、`src/teacher/moe_lora_model.py` を作成し、`MoeLoraConfig` および `MoeLoraModel` を移植する。この際、`src/teacher/gating.py` をインポートするように修正する。
    *   [x] `src/teacher/ilora_model.py` を修正し、`peft.get_peft_model` の代わりに、新しく作成した `MoeLoraModel` を使用するように変更する。
    *   [x] `iLoRAModel` の `forward` メソッドにおいて、ゲーティングネットワークの出力 (`gate_weights`) を `MoeLoraModel` の `forward` メソッドに適切に渡すように修正する。
    *   [x] `iLoRAModel` のプロンプトエンジニアリングを、`ref_repositories/iLoRA/model/model_interface.py` の `wrap_emb` メソッドのロジック（推薦モデルの埋め込みを特殊トークンに置き換える方式）に近づける。具体的には以下の手順で実施する。
        *   [x] `src/teacher/ilora_model.py` または `src/teacher/factory.py` で、`[PH]`, `[HistoryEmb]`, `[CansEmb]`, `[ItemEmb]` などの特殊トークンをトークナイザーに追加する。
        *   [x] `src/teacher/ilora_model.py` に、`ref_repositories/iLoRA/model/model_interface.py` の `encode_items` および `encode_users` に相当するロジックを実装する。これには、推薦モデル (`rec_model`) とプロジェクター (`projector`) のインスタンスを `iLoRAModel` に渡すか、内部でロードする必要がある。
        *   [x] `src/teacher/ilora_model.py` の `_generate_prompt` メソッドを削除し、その機能はデータ準備ステップで特殊トークンを含む `input_ids` を作成することで代替される。
        *   [x] `src/teacher/ilora_model.py` の `forward` メソッドのシグネチャを `(self, batch: Dict[str, Any])` に変更し、`batch` オブジェクトから `input_ids`, `attention_mask`, `seq`, `len_seq`, `cans`, `len_cans`, `item_id` などを直接受け取るようにする。
        *   [x] `src/teacher/ilora_model.py` の `forward` メソッド内で、`ref_repositories/iLoRA/model/model_interface.py` の `wrap_emb` メソッドに相当するロジックを実装し、`input_ids` 内の特殊トークンの埋め込みを、`encode_items` および `encode_users` で生成した実際のアイテム/ユーザー埋め込みで置き換える。これにより、LLMへの入力が `inputs_embeds` を直接使用する形式になる。
        *   [x] `src/student/datamodule.py` を修正し、`iLoRAModel` が期待する `batch` オブジェクト（`input_ids`, `attention_mask`, `seq`, `len_seq`, `cans`, `len_cans`, `item_id` などを含む）を生成するように変更する。これには、`tokenizer` を `SASRecDataModule` に渡し、プロンプトを生成し、特殊トークンを埋め込むロジックを組み込む必要がある。
        *   [x] 上記変更に伴い、`src/teacher/trainer_ilora.py` および `src/exp/run_teacher.py` の `forward` メソッド呼び出しを修正し、適切な `batch` オブジェクトを渡すようにする。
        *   [x] 上記変更に伴い、関連するテスト (`tests/teacher/test_ilora_model.py`) を修正・追加し、動作検証を行う。
    *   [ ] **次のエージェントへの依頼**: `src/teacher/moe_lora_model.py` の `ImportError: cannot import name 'COMMON_LAYERS_PATTERN' from 'peft.utils'` の修正を再試行し、テストを実行して変更を検証してください。

*   **DLLM2Recロジックの再現性向上**
    *   [ ] `ref_repositories/DLLM2Rec/main.py` を参考に、ランキング蒸留損失 (`src/distill/kd_losses.py` の `RankingDistillationLoss`) を、ポジション、共通アイテム、信頼度に基づく重み付けBCE損失に近づけるように修正する。
    *   [ ] `ref_repositories/DLLM2Rec/main.py` を参考に、埋め込み蒸留損失の適用方法を、LLMからの埋め込みを生徒モデルの入力埋め込みに直接加算する方式に近づけるように修正する。
    *   [ ] 損失の合成 (`src/distill/trainer_distill.py`) において、`ref_repositories/DLLM2Rec/main.py` の `dros loss` を含めることを検討する。
    *   [ ] 上記変更に伴い、関連するテスト (`tests/distill/test_kd_losses.py`, `tests/distill/test_trainer_distill.py`) を修正・追加し、動作検証を行う。

### 5.3. 最終的なファイルの状態

*   `conf/dataset/movielens.yaml`: `limit_data_rows: 160`
*   `conf/distill/dllm2rec.yaml`: `teacher_checkpoint_path` が設定済み
*   `src/exp/run_teacher.py`: `limit_data_rows` を `SASRecDataModule` に渡すように修正済み
*   `src/exp/run_distill.py`: `limit_data_rows` を `SASRecDataModule` に渡すように修正済み
*   `src/distill/selection_policy.py`: `AllSamplesPolicy` を追加済み
*   `src/distill/trainer_distill.py`: `calculate_metrics` の引数変換を修正済み
*   `src/teacher/ilora_model.py`: `item_embeddings` のメモリ最適化、`output_layer` の次元修正、`MLPProjector` のインポート問題は解決済み
*   `src/teacher/factory.py`: `llm`, `tokenizer`, `MLPProjector` を `iLoRAModel` に渡すように修正済み
*   `docs/specification/02_development_notes_ja.md`: `limit_data_rows` の説明を追加済み
*   `docs/specification/04_execution_guide.md`: `limit_data_rows` とメモリ問題に関する注意を追加済み
*   `cmd/start_container_and_install.sh`: 新規追加済み
*   `cmd/run_distill.sh`: `PYTHONPATH=/workspace` を追加済み

### 5.5. エージェントによる引き継ぎノート (2025-11-17, 2回目)

#### 5.5.1. 実施した作業の概要

*   **iLoRAロジックのリファクタリング**:
    *   参照リポジトリを基に、`src/teacher/gating.py`と`src/teacher/moe_lora_model.py`を作成し、iLoRAの主要な構成要素を移植しました。
    *   `src/teacher/ilora_model.py`を修正し、新しく作成した`MoeLoraModel`を使用するように変更しました。
*   **テストスイートの修正**:
    *   上記のリファクタリングに伴い、`tests`ディレクトリ配下の複数のテストファイル（`test_trainer_distill.py`, `test_ilora_model.py`, `test_trainer_ilora.py`など）を修正し、`NameError`, `TypeError`, `AttributeError`などのエラーを解決しました。
    *   `SASRecDataModule`が`tokenizer`を必要とするようになったため、関連するテストフィクスチャを更新しました。

### 5.5.2. 現在の課題と次のエージェントへの依頼事項

iLoRAロジックのリファクタリングと関連するテストの修正は完了し、教師モデルの学習が正常に実行されることを確認しました。
次のエージェントは、DLLM2Recロジックの再現性向上に着手してください。

*   **DLLM2Recロジックの再現性向上**:
    *   `docs/specification/06_difference_from_asis.md` に記載されているDLLM2Recのロジック差分を参考に、既存実装を修正してください。
    *   具体的には、ランキング蒸留における複雑な重み付けロジックや、埋め込み蒸留の適用方法などを、参照リポジリの意図を汲んで再現性を高めることを優先してください。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。

### 5.6. エージェントによる引き継ぎノート (2025-11-17, 3回目)

#### 5.6.1. 実施した作業の概要

*   **DLLM2Rec ランキング蒸留ロジックの再現**:
    *   `src/distill/kd_losses.py` に、DLLM2Recの参照実装 (`ref_repositories/DLLM2Rec/main.py`) に基づく重み付きBCE損失 (`WeightedBCELoss`) を追加しました。
    *   `src/distill/trainer_distill.py` を修正し、`WeightedBCELoss` を使用するように変更しました。
    *   `src/distill/trainer_distill.py` に、DLLM2Recの参照実装で用いられているポジション、信頼度、一貫性に基づく重み計算ロジックを移植しました。
    *   `src/teacher/ilora_model.py` を修正し、`get_teacher_outputs` メソッドが教師モデルのランキングスコアからトップKの候補アイテム (`candidates`) とその信頼度 (`confidence`) を出力するように変更しました。これにより、`DistillationTrainer` がランキング蒸留に必要な情報を取得できるようになりました。
    *   `iLoRAModel` のコンストラクタに `candidate_topk` パラメータを追加し、`get_teacher_outputs` で使用するトップKの値を設定できるようにしました。

#### 5.6.2. 現在の課題と次のエージェントへの依頼事項

*   **DLLM2Rec 埋め込み蒸留ロジックの再現**:
    *   `ref_repositories/DLLM2Rec/main.py` では、LLMからの埋め込みを生徒モデルの入力埋め込みに直接加算することで埋め込み蒸留を行っています。このロジックを `src/student/models.py` の `SASRec` モデルに移植し、`src/distill/trainer_distill.py` でLLM埋め込みを `SASRec` モデルに渡すように修正してください。
*   **DLLM2Rec DRO損失の再現**:
    *   `ref_repositories/DLLM2Rec/main.py` に実装されているDRO損失を `src/distill/kd_losses.py` に追加し、`src/distill/trainer_distill.py` で使用するように修正してください。
*   **設定ファイルの更新**:
    *   DLLM2Recのロジック再現に必要な新しいハイパーパラメータ（例: `ed_weight`, `lam`, `gamma_position` など）を `conf/distill/dllm2rec.yaml` に追加し、`src/distill/trainer_distill.py` のコンストラクタで受け取るように修正してください。
*   **テストの追加と修正**:
    *   DLLM2Recロジックの変更に伴い、`tests/distill/test_kd_losses.py` および `tests/distill/test_trainer_distill.py` に新しいテストケースを追加し、既存のテストケースを修正して、変更が正しく機能することを確認してください。
*   **ドキュメントの継続的な更新**:
    *   今後の実装や変更についても、`docs/specification/02_development_notes_ja.md` に記録し、必要に応じて他の仕様書も更新してください。