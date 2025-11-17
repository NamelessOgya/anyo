# 実装仕様書（コンテナ設計＋docs構成込み・決定版）

**LLM → シーケンシャル推薦モデル蒸留基盤**
（iLoRA ロジック + DLLM2Rec ロジックを自前実装 / Hydra + Poetry / Dockerコンテナ）

---

## 0. ドキュメントの目的

* この仕様書は、過去の経緯を知らない実装者・エージェントに渡す前提の**唯一の設計書**です。
* これを読めば以下が分かることを目標とします：

  * プロジェクト構造（ディレクトリ・docs構成含む）
  * コンテナ設計
  * Python / Poetry / Hydra 環境
  * 教師モデル（iLoRA ロジック）の実装方針
  * 蒸留（DLLM2Rec ロジック）の実装方針
* さらに、**docs ディレクトリの使い方**を明確にします：

  * `docs/implement.md`：実装メモ（日本語、進捗・TODO・注意点を書く）
  * `docs/specification/*.md`：仕様書・設計書を Markdown 形式で保存する場所（本仕様書も含めて）

---

## 1. ゴールと基本方針

### 1.1 ゴール

* **教師モデル（LLM レコメンダ）**

  * iLoRA（Instance-wise LoRA）の**処理ロジック（アルゴリズム）を再現した** LLM ベースのシーケンシャルレコメンダを実装する。
  * コードは自前実装だが、「どう動くか」は iLoRA 論文・公式実装と同等を目指す。
  * 想定GPU：A100 1枚で学習・推論可能。

* **生徒モデル**

  * SASRec などの軽量なシーケンシャル推薦モデル（ID ベース）。

* **知識蒸留**

  * DLLM2Rec の論文・実装を参考に、

    * ランキング蒸留
    * 埋め込み蒸留
      を組み合わせた **DLLM2Rec 相当ロジック**を自前実装する。

* **実験管理**

  * 依存管理：Poetry
  * 設定管理：Hydra
  * 実験環境：Docker コンテナ（起動スクリプトは `env_shell` 配下）
  * 実験結果：Hydra の `run.dir` を `result/result_{timestamp}` に設定し、その下に config・モデル・ログ・メトリクスを保存。

* **将来拡張**

  * Active Learning / Meta Learning による「どのサンプルを蒸留に使うか」を `SelectionPolicy` インターフェース経由で追加できるよう設計（初期実装は「全サンプル使用」のみ）。

### 1.2 参考リポジトリとの関係

* 参考（コードはコピペ禁止）：

  * iLoRA

    * 論文: *Customizing Language Models with Instance-wise LoRA for Sequential Recommendation*
    * GitHub: [https://github.com/AkaliKong/iLoRA](https://github.com/AkaliKong/iLoRA)
  * DLLM2Rec

    * GitHub: [https://github.com/istarryn/DLLM2Rec](https://github.com/istarryn/DLLM2Rec)

→ これらは**挙動・データ形式・アルゴリズムの参考資料**としてのみ使い、
実装は本リポジトリ内で新規に書く。

---

## 2. 技術スタックと環境

### 2.1 言語・ランタイム

* Python バージョン：**固定しない**（開発環境に合わせる / 3.9〜3.12 程度を想定）
* Poetry の `python` 制約は例として `>=3.9,<4.0` 程度に広めでよい。

### 2.2 ライブラリ

* PyTorch
* Transformers
* Hydra (1.3系)
* TensorBoard
* numpy / pandas
* LoRA 実装用に `peft` など（必要に応じて）

### 2.3 実験環境

* 実験は **Docker コンテナ内で実行する前提**。
* コンテナ起動・接続は `env_shell` 配下のスクリプトを経由する。

---

## 3. リポジトリ構成

```text
.
├── Dockerfile                     # 実験用コンテナのビルド定義
├── pyproject.toml                 # Poetry 設定（Pythonバージョンは緩め）
├── poetry.lock                    # 依存ロック（必ずコミット）
├── src/                           # すべての Python ソース
├── conf/                          # Hydra コンフィグ
├── cmd/                           # 実験用 .sh スクリプト（Poetry + Hydra 呼び出し）
├── env_shell/                     # 環境設定・コンテナ起動用スクリプト
├── data/                          # 生データ・前処理データ・教師出力など
├── result/                        # 実験結果（Hydra run.dir）
├── ref_repositories/              # 参考用の外部リポジトリ（import禁止）
│   ├── ilora/
│   └── dllm2rec/
└── docs/
    ├── implement.md               # 実装メモ（日本語, 実装状況/注意点を記録）
    └── specification/             # 仕様書・設計書置き場（Markdown）
        ├── 00_overview.md         # ← 本仕様書をここに保存することを推奨（例）
        ├── 01_container_env.md    # 他の仕様書を分割する場合の例
        └── ...                    # 必要に応じて追加
```

### 3.1 docs ディレクトリの運用ルール（★重要）

* `docs/implement.md`

  * 実装者が進捗や実験メモを**日本語で**書き残すファイル。
  * 「どこまで実装済みか」「何がハマりポイントか」「次の担当者への注意」などを記録すること。

* `docs/specification/`

  * 仕様書・設計書を Markdown 形式 (`.md`) で配置する場所。
  * ファイル名は任意だが、**適切な粒度で分割してよい**：

    * 例：

      * `00_overview.md`（全体仕様）
      * `01_container_and_env.md`（コンテナ・環境構築詳細）
      * `02_teacher_ilora_logic.md`（教師 iLoRA 部分の詳細）
      * `03_distill_dllm2rec_logic.md`（蒸留ロジック詳細）
  * **命令**：

    * この仕様書（いま読んでいる内容を整理したもの）も含めて、
      仕様書・設計書は **必ず `docs/specification/` 配下に `.md` 形式で保存すること**。
    * 将来的に仕様を追加・分割するときも、必ず `docs/specification/` に Markdown ファイルとして追記すること。

---

## 4. コンテナ設計

### 4.1 目的

* GPU対応・Poetry対応の実験環境を Docker コンテナで統一し、
  「コンテナに入れば同じ方法で実験を再現できる」状態にする。

### 4.2 Dockerfile の要件

* ルート直下に `Dockerfile` を配置。
* 要件：

  1. ベースイメージ：

     * 例：`pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel`
  2. OS パッケージ：

     * git, curl, build-essential 等
  3. Poetry のインストール：

     * `pip install "poetry>=1.7"` 等で利用可能にする。
  4. 作業ディレクトリ：

     * `WORKDIR /workspace`
     * ソースコードはホストから `-v` マウントするので、Dockerfile 内で `COPY` は不要。

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=1.7"

WORKDIR /workspace
```

### 4.3 env_shell 配下のコンテナ起動スクリプト

`env_shell/start_experiment_container.sh` を必ず用意し、
以下のような骨子で実装すること：

```bash
#!/usr/bin/env bash
set -e

CONTAINER_NAME="anyo-experiment"
HOST_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="anyo-ilora-dllm2rec:latest"

# イメージがなければビルド
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1;
then
  docker build -t "${IMAGE_NAME}" "${HOST_PROJECT_ROOT}"
fi

docker run -it --rm \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  -v "${HOST_PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash
```

* 実験は基本的に：

  1. ホストで `env_shell/start_experiment_container.sh` を実行してコンテナに入る。
  2. コンテナ内で `poetry install` → `poetry run ...` で実験を回す。

---

## 5. Poetry & 実行方法（コンテナ内）

### 5.1 Poetry 設定（pyproject.toml のイメージ）

```toml
[tool.poetry]
name = "anyo-ilora-dllm2rec"
version = "0.1.0"
description = "LLM->Sequential Rec distillation framework (iLoRA logic + DLLM2Rec logic)"
authors = ["<Your Name> <you@example.com>"]
package-mode = false

[tool.poetry.dependencies]
python = ">=3.9,<4.0"
torch = "^2.3"
transformers = "^4.43"
hydra-core = "^1.3"
numpy = "^1.26"
pandas = "^2.2"
tensorboard = "^2.18"
peft = "^0.12"

[tool.poetry.group.dev.dependencies]
pytest = "^8"
black = "^24"
ruff = "^0.7"

[tool.poetry.scripts]
run-teacher = "src.exp.run_teacher:main"
run-student-baseline = "src.exp.run_student_baseline:main"
run-distill = "src.exp.run_distill:main"
run-eval-all = "src.exp.run_eval_all:main"
```

### 5.2 実行例（コンテナ内）

```bash
poetry install

poetry run python -m src.exp.run_teacher
poetry run python -m src.exp.run_student_baseline
poetry run python -m src.exp.run_distill
poetry run python -m src.exp.run_eval_all
```

---

## 6. Hydra コンフィグ（概要）

* `conf/config.yaml` で defaults を定義し、
* `conf/dataset/*.yaml`, `conf/teacher/*.yaml`, `conf/student/*.yaml`, `conf/distill/*.yaml`, `conf/active/*.yaml` で詳細を分割。
* `conf/hydra/default.yaml`：

```yaml
hydra:
  run:
    dir: "result/result_${now:%Y%m%d_%H%M%S}"
  job:
    chdir: true
```

→ 実験ごとに `result/result_YYYYMMDD_HHMMSS/` に結果一式（.hydra, logs, models, metrics）が保存される。

---

## 7. `src/` 構成と役割（要点）

```text
src
├── core/                 # 共通ユーティリティ
│   ├── paths.py
│   ├── logging.py
│   ├── metrics.py
│   ├── data_utils.py
│   ├── seed.py
│   └── git_info.py
├── teacher/              # 教師モデル（iLoRAロジック）
│   ├── interfaces.py
│   ├── ilora_model.py
│   └── factory.py
├── student/              # 生徒モデル（SASRecなど）
│   ├── models.py
│   ├── datamodule.py
│   ├── trainer_baseline.py
│   └── evaluator.py
├── distill/              # DLLM2Recロジックの蒸留
│   ├── data_bridge.py
│   ├── kd_losses.py
│   ├── selection_policy.py
│   └── trainer_distill.py
└── exp/                  # 実験用エントリポイント
    ├── run_teacher.py
    ├── run_student_baseline.py
    ├── run_distill.py
    └── run_eval_all.py
```

* 詳細なロジック（iLoRA のゲーティング・LoRA合成, DLLM2Rec の KD loss など）は、
  各ファイル内の docstring や、必要に応じて `docs/specification/*` に詳細仕様を書き足していくこと。

---

## 8. iLoRA / DLLM2Rec ロジック再現の方針（超要約）

* **iLoRA（teacher/ilora_model.py）**

  * LLM + 複数 LoRA エキスパート + シーケンスごとのゲーティング。
  * `h_seq`（シーケンス表現） → softmaxゲート → LoRAパラメータを線形結合 → LLMに適用。
  * 学習：次アイテム予測（CE loss）、学習対象は LoRA + ゲートのみ。
  * 教師出力として DLLM2Rec 互換のランキング・スコア・埋め込みを出す。

* **DLLM2Rec（distill/）**

  * 教師ランキング＋スコアを用いたランキング蒸留 loss。
  * 教師埋め込み vs 生徒埋め込みの距離を縮める埋め込み蒸留 loss。
  * CE loss（通常学習）＋ KD loss の合成。

詳細な数式や実装戦略は、必要に応じて
`docs/specification/02_teacher_ilora_logic.md` や `03_distill_dllm2rec_logic.md` といったファイルに掘り下げて記述してください。

---

## 9. 評価・時間計測

* 指標：

  * Recall@K / NDCG@K / HitRatio@K（K は Hydra config で指定）
  * 比較対象：

    * 生徒ベースライン
    * 蒸留あり（DLLM2Rec ロジック）

* 時間計測（`metrics/time.json`）：

  * `teacher_train_time_total`
  * `distill_teacher_export_time`
  * `student_baseline_train_time`
  * `distill_student_train_time`

---

## 10. docs/implement.md の運用（必須）

* パス：`docs/implement.md`
* 言語：**日本語**
* 内容（例）：

```markdown
# ANYO LLM→Rec 蒸留基盤 実装メモ

## 1. 現在までの実装状況
- core/ : ログ・メトリクス・time計測まで実装済み
- teacher/ilora_model.py : LoRAエキスパート定義まで、ゲート合成はTODO
- student/models.py : SASRec 実装済み, Movielensで簡易テスト済み
- distill/kd_losses.py : ランキング蒸留のみ実装, embedding蒸留は未完 など

## 2. コンテナ・環境
- Python: 3.x
- イメージ: anyo-ilora-dllm2rec:latest（Dockerfileからビルド）
- 起動: `env_shell/start_experiment_container.sh`

## 3. 実行方法メモ
- 教師学習: `poetry run python -m src.exp.run_teacher dataset=movielens`
- 生徒ベースライン: `poetry run python -m src.exp.run_student_baseline dataset=movielens`
- 蒸留: `poetry run python -m src.exp.run_distill dataset=movielens`

## 4. 既知の問題 / TODO
- 例: 「iLoRAのプロンプト設計は暫定、要改善」
- 例: 「Amazon Games データセットは未検証」

## 5. 次の実装者へのコメント
- 例: 「まず run_teacher を動かして teacher_outputs の形式を確認すると理解しやすいです」
- 例: 「DLLM2Rec のリポジトリの data_reader 周りを読むと data_bridge の設計に役立ちます」
```

**命令：**

* 実装者は作業の節目ごとに `docs/implement.md` を更新し、
  「どこまで終わっていて何が残っているか」が分かるようにしてください。

---

## 11. 仕様書ファイルの出力ルール（再掲）

* 本仕様書を含め、**仕様・設計に関する Markdown はすべて `docs/specification/` 配下に置くこと**。
* ファイル名は任意だが、章ごと・関心ごとに分割してよい。
* このチャットで記述した内容を反映したベース仕様は、例えば：

  * `docs/specification/00_overview.md`

  として保存してください（実際のファイル名は任意ですが、このように整理しておくと分かりやすいです）。
