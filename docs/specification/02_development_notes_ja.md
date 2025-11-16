# 開発上の注意点と将来の考慮事項

このドキュメントは、本プロジェクトの開発における潜在的な改善点、未着手のタスク、および注意点をまとめたものです。`00_overview.md` を読んだ後に参照してください。

---

## 1. 実行環境とコンテナ (`env_shell`)

### 1.1. 開発体験の向上

現在の `start_experiment_container.sh` スクリプトは、コンテナ内で `bash` セッションを開始します。その後、開発者は手動で `poetry install` を実行する必要があります。これは、よりスムーズな開発導入体験のために改善できます。

**推奨事項:**

*   以下の機能を持つ `entrypoint.sh` スクリプトを作成する：
    1.  `poetry.lock` がインストール済みの依存関係より新しいかどうかを確認する。
    2.  必要であれば `poetry install` を実行する。
    3.  コンテナに渡されたコマンド（例：`bash` やPythonスクリプト）を実行する。
*   `start_experiment_container.sh` の `docker run` コマンドを修正し、この `entrypoint.sh` を使用するように変更する。

### 1.2. ファイルのパーミッション問題 (UID/GID)

マウントされた `/workspace` ボリューム内でコンテナ内でファイルが作成されると、それらは `root` ユーザー（またはコンテナの実行ユーザー）によって所有されます。これにより、ホストマシン上でパーミッションの問題が発生する可能性があります。

**推奨事項:**

*   `start_experiment_container.sh` を修正し、ホストユーザーのUIDとGIDをコンテナに渡すようにする。
*   `Dockerfile` または `entrypoint.sh` で、同じUID/GIDを持つユーザーをコンテナ内に作成し、そのユーザーとしてコマンドを実行するようにする。

**`start_experiment_container.sh` のスニペット例:**

```bash
docker run -it --rm \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  -v "${HOST_PROJECT_ROOT}:/workspace" \
  -w /workspace \
  -e "HOST_UID=$(id -u)" \
  -e "HOST_GID=$(id -g)" \
  "${IMAGE_NAME}" \
  bash
```

---

## 2. 設定ファイル (`conf`)

### 2.1. プレースホルダーのデフォルト値

メインの `conf/config.yaml` は、`defaults` リストにプレースホルダーとして `???` を含んでいます。これらは実際のコンフィグファイルに置き換える必要があります。

**TODO:**

*   各グループのデフォルトコンフィグファイルを作成する。例：
    *   `conf/dataset/movielens.yaml`
    *   `conf/teacher/ilora_default.yaml`
    *   `conf/student/sasrec_default.yaml`
    *   `conf/distill/dllm2rec_default.yaml`
    *   `conf/active/full_data.yaml`
*   初期テストのために、`conf/config.yaml` を更新して、これらのいずれかをデフォルトとして使用するようにする。

### 2.2. 未定義のコンフィグパラメータ

各コンフィグファイル（学習率、モデルの次元、データセットのパスなど）のスキーマとパラメータはまだ定義されていません。

**TODO:**

*   各コンポーネント（`teacher`, `student`, `distill` など）に必要なパラメータを、それぞれのYAMLファイルで定義する。これは、対応する `src` モジュールの実装前または実装中に行うべきです。

---

## 3. 依存関係 (`pyproject.toml`)

依存関係は `^` で指定されており、マイナーバージョンの更新が可能です（例：`^1.3` は `1.4` をインストールする可能性があります）。`poetry.lock` が再現性を保証しますが、この点には注意が必要です。

**注意点:**

*   依存関係を変更（`poetry add`, `poetry install`, `poetry update`）した後は、必ず `poetry.lock` ファイルをコミットしてください。
*   非常に厳密な環境を求める場合は、正確なバージョンを固定する（例：`torch = "2.3.0"`）ことを検討してください。しかし、一般的な開発では現在の方法で問題ありません。

---

## 4. ソースコードの実装 (`src`)

`src` ディレクトリには現在、空のスケルトンファイルしか含まれていません。

**推奨される実装順序:**

1.  **`src/core`**: 最初に共通のユーティリティを実装します。`paths.py`, `seed.py`, `logging.py` から始めるのが良いでしょう。
2.  **`src/student`**: ベースラインとなる生徒モデル（例：SASRec）を実装します。これは通常、教師モデルよりも単純であり、データ読み込みと評価パイプラインを検証するために使用できます。
    *   `datamodule.py`: データ読み込み用。
    *   `models.py`: SASRecモデル本体。
    *   `evaluator.py`: メトリクス計算用。
    *   `trainer_baseline.py`: ベースラインの学習ループ。
    *   `exp/run_student_baseline.py`: ベースライン実験を実行するためのエントリポイント。
3.  **`src/teacher`**: iLoRAのロジックを実装します。これは最も複雑な部分です。
4.  **`src/distill`**: 教師モデルと生徒モデルの両方に依存する蒸留ロジックを実装します。

---

## 5. テスト

プロジェクトには `pytest` が設定されていますが、テストの構造はまだありません。

**推奨事項:**

*   プロジェクトルートに `tests/` ディレクトリを作成します。
*   `tests/` 内に `src/` の構造をミラーリングします。例えば、`src/core/paths.py` のテストは `tests/core/test_paths.py` に配置します。
*   テストを簡単に実行するためのスクリプトまたは `[tool.poetry.scripts]` エントリを追加します（例：`poetry run pytest`）。
*   まず、`core` ユーティリティの簡単な単体テストから書き始めることを推奨します。
*   データ読み込みやモデルコンポーネントが開発されたら、それらのテストを記述します。

---

## 6. 現在までの実装状況

### `src/core` の実装状況:

*   `paths.py`: プロジェクト内の主要なパスを取得するユーティリティを実装済み。
*   `seed.py`: 実験の再現性を確保するための乱数シード設定ユーティリティを実装済み。
*   `logging.py`: コンソールとファイルへのロギング設定ユーティリティを実装済み。
*   `git_info.py`: 現在のGitリポジリ情報を取得するユーティリティを実装済み。
*   `metrics.py`: 推薦モデルの評価指標（Recall@K, NDCG@K, HitRatio@K）を計算するユーティリティを実装済み。
*   `data_utils.py`: データ処理に関する一般的なユーティリティ（現状は空ファイル、必要に応じて追加予定）。

### `src/student` の実装状況:

*   `models.py`: SASRec (Self-Attentive Sequential Recommendation) モデルを実装済み。
*   `datamodule.py`: SASRecモデル用のデータセット (`SASRecDataset`) とデータローダー (`SASRecDataModule`) を実装済み。`ref_repositories/iLoRA/data/ref` からMovielensデータを読み込み、SASRecの入力形式に整形します。
*   `trainer_baseline.py`: SASRecモデルの学習ロジックをカプセル化する `pytorch_lightning.LightningModule` を実装済み。
*   `evaluator.py`: 学習済みSASRecモデルの評価を行うユーティリティを実装済み。

### `src/teacher` の実装状況:

*   `interfaces.py`: 教師モデルが満たすべきインターフェース (`TeacherModel`) を定義済み。
*   `ilora_model.py`: iLoRA (Instance-wise LoRA) ロジックを再現したLLMベースのシーケンシャルレコメンダのスケルトンを実装済み。LLMのロード、LoRAアダプターの準備、ゲーティングメカニズムの基本的な構造を含みます。
*   `trainer_ilora.py`: `iLoRAModel` をラップし、学習ロジックをカプセル化する `pytorch_lightning.LightningModule` を実装済み。
*   `factory.py`: Hydraの設定に基づいて教師モデルのインスタンスを生成するファクトリ関数 (`create_teacher_model`) を実装済み。

### `src/distill` の実装状況:

*   `kd_losses.py`: ランキング蒸留損失 (`RankingDistillationLoss`) と埋め込み蒸留損失 (`EmbeddingDistillationLoss`) を実装済み。
*   `selection_policy.py`: 蒸留に用いるサンプルを選択するポリシーの抽象基底クラス (`SelectionPolicy`) と、すべてのサンプルを使用する具象クラス (`AllSamplesPolicy`) を実装済み。
*   `data_bridge.py`: 教師モデルと生徒モデルの出力を蒸留損失計算に適した形式に橋渡しするクラス (`DataBridge`) を実装済み。
*   `trainer_distill.py`: 知識蒸留の学習ロジックをカプセル化する `pytorch_lightning.LightningModule` (`DistillationTrainer`) を実装済み。

### `src/exp` の実装状況:

*   `run_teacher.py`: iLoRA教師モデルの学習と評価を実行するためのエントリポイントスクリプトを実装済み。
*   `run_student_baseline.py`: SASRec生徒モデルのベースライン学習と評価を実行するためのエントリポイントスクリプトを実装済み。
*   `run_distill.py`: 知識蒸留の学習と評価を実行するためのエントリポイントスクリプトを実装済み。
*   `run_eval_all.py`: 複数の学習済みモデルをまとめて評価するためのエントリポイントスクリプトを実装済み。