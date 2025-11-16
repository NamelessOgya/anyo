了解です！
では、「**コンテナ設計まで含めて、これだけ読めば誰でも実装に着手できる**」仕様書として、もう一度フルでまとめ直します。

---

# 実装仕様書（コンテナ設計込み・決定版）

**LLM → シーケンシャル推薦モデル蒸留基盤**
（iLoRA ロジック + DLLM2Rec ロジックを自前実装 / Hydra + Poetry / コンテナ利用）

---

## 0. ドキュメントの目的

* この仕様書は、
  **過去の会話や背景を知らない実装者 / コーディングエージェントに渡す前提**の「唯一の設計書」です。
* これを読めば：

  * プロジェクト構造
  * コンテナ設計
  * Python / Poetry / Hydra の前提
  * 教師モデル（iLoRA ロジック）の実装方針
  * 蒸留（DLLM2Rec ロジック）の実装方針
* さらに、実装者は **作業メモを `discussions/docs/specification.md` に日本語で残す義務**があります（§10）。

---

## 1. ゴールと基本方針

### 1.1 ゴール

* **教師モデル（LLM レコメンダ）**

  * iLoRA（Instance-wise LoRA）の**処理ロジック（アルゴリズム）を再現**した LLM ベースのシーケンシャルレコメンダを実装する。
  * コードは自前実装だが、「どう動くか」は iLoRA 論文・公式実装と同等を目指す。
  * 想定 GPU：A100 1枚で学習・推論可能。

* **生徒モデル**

  * SASRec などの軽量シーケンシャル推薦モデル（IDベース）。

* **知識蒸留**

  * DLLM2Rec の論文・実装を参考に、

    * ランキング蒸留
    * 埋め込み蒸留
      を組み合わせた **DLLM2Rec 相当ロジック**を自前実装する。

* **実験管理**

  * 依存管理：Poetry
  * 設定管理：Hydra
  * 実験環境：Docker コンテナ（起動スクリプトは `env_shell` 配下に配置）
  * 実験結果：`result/result_{timestamp}` 配下に集約（Hydra run.dir）

* **将来拡張**

  * Active Learning / Meta Learning による「どのサンプルを蒸留に使うか」の選択を、
    `SelectionPolicy` インターフェース経由で後から追加できるように設計する（現段階では全サンプル使用のみ）。

### 1.2 参考リポジトリとの関係

* 参考として読むが、コードはコピペしない：

  * iLoRA

    * 論文: *Customizing Language Models with Instance-wise LoRA for Sequential Recommendation*
    * GitHub: [https://github.com/AkaliKong/iLoRA](https://github.com/AkaliKong/iLoRA)
  * DLLM2Rec

    * GitHub: [https://github.com/istarryn/DLLM2Rec](https://github.com/istarryn/DLLM2Rec)

* これらは「挙動・データ形式・アルゴリズム」を理解するための**参照資料**としてのみ利用し、
  実装はすべて本リポジトリ内で新規に書く。

---

## 2. 技術スタックと環境

### 2.1 言語・ランタイム

* Python バージョン：**固定しない**（開発環境に合わせてよい）

  * ただし、Poetry の `python` 制約は `>=3.9,<4.0` 程度に広めに設定すること。

### 2.2 主要ライブラリ

* PyTorch
* Transformers
* Hydra (1.3系)
* TensorBoard
* numpy / pandas
* LoRA 実装用：`peft` など（必要に応じて）

### 2.3 コンテナ利用が前提

* 実験は **Docker コンテナ内で実行することを前提**とする。
* 起動・停止・接続などの操作は、必ず `env_shell` 配下のスクリプト経由で行う。

---

## 3. リポジトリ構成

```text
.
├── Dockerfile                     # 実験用コンテナのビルド定義（後述）
├── pyproject.toml                 # Poetry 設定（Pythonバージョンは緩め）
├── poetry.lock                    # 依存ロック（必ずコミット）
├── src/                           # すべての Python ソース
├── conf/                          # Hydra コンフィグ
├── cmd/                           # 実験用 .sh スクリプト（Poetry + Hydra 呼び出し）
├── env_shell/                     # 環境設定・コンテナ起動用スクリプト（★重要）
├── data/                          # 生データ・前処理データ・教師出力など
├── result/                        # 実験結果（Hydra run.dir）
├── ref_repositories/              # 参考用の外部リポジトリ（import禁止）
│   ├── ilora/
│   └── dllm2rec/
└── discussions/
    └── docs/
        └── specification.md      # 実装メモ（日本語、必須）
```

---

## 4. コンテナ設計（★この章が今回追加）

### 4.1 ゴール

* 誰がどこで実験しても、以下が満たされるようなコンテナを設計・利用する：

  * GPU（A100）対応
  * PyTorch / Transformers / Hydra / Poetry が利用可能
  * ホストのソースコード・データをコンテナにマウントして利用
  * コンテナ内からは、**Poetry 経由で**実験スクリプト (`src/exp/*.py`) を実行

### 4.2 Dockerfile の要件

ルート直下に `Dockerfile` を作成する。要件は以下：

1. **ベースイメージ**

   * GPU対応の PyTorch 公式イメージを推奨：

     * 例：`pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel`
   * これにより CUDA / cuDNN / PyTorch 周りは基本済になっている前提で進められる。

2. **必要パッケージの導入**

   * OS パッケージ：

     * git
     * curl
     * python3-venv（必要に応じて）
     * build-essential など
   * Poetry のインストール：

     * `pip install poetry` もしくは `pipx` 経由（どちらでも可 / 仕様としては「Poetry が使えること」が条件）。

3. **作業ディレクトリ**

   * `/workspace` をコンテナ内の作業ディレクトリとして設定。
   * ソースコードはホストからマウントするため、Dockerfile 内で `COPY` はしない（ビルド時ではなくランタイム時に mount）。

4. **エントリポイント**

   * エントリポイントは特に固定しなくてよい（`bash` など）。
   * 実際の実験実行は `env_shell` の起動スクリプト経由で `poetry run ...` を叩く。

#### Dockerfile のイメージ（ざっくり例）

※ 実装者はこれをベースに具体的な Dockerfile を書いてください。

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn9-devel

# 基本ツール
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetry インストール
RUN pip install --no-cache-dir "poetry>=1.7"

# 作業ディレクトリ
WORKDIR /workspace

# ここではコードをCOPYしない。ホストから -v マウントする想定。
```

### 4.3 env_shell 配下のコンテナ起動スクリプト

`env_shell/` に以下のようなスクリプトを用意すること：

#### 4.3.1 `env_shell/start_experiment_container.sh`

目的：

* Docker コンテナを起動し、ホストのリポジトリを `/workspace` にマウントする。
* GPU を利用可能にする。

想定内容（骨子）：

```bash
#!/usr/bin/env bash
set -e

# コンテナ名（任意）
CONTAINER_NAME="anyo-experiment"

# ホスト側のプロジェクトルート
HOST_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 使用するイメージ名（Dockerfile からビルドしたもの）
IMAGE_NAME="anyo-ilora-dllm2rec:latest"

# まだイメージがなければビルド
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
  docker build -t "${IMAGE_NAME}" "${HOST_PROJECT_ROOT}"
fi

# コンテナ起動（すでに起動済みなら attach でもよい）
docker run -it --rm \
  --gpus all \
  --name "${CONTAINER_NAME}" \
  -v "${HOST_PROJECT_ROOT}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash
```

ポイント：

* **実験は常にこのスクリプト経由でコンテナを立ち上げて行う**。
* VS Code Remote-Containers 等を使う場合は、その設定でも `/workspace` を前提にしてよい。

#### 4.3.2 その他のスクリプト（任意）

必要なら：

* attach 用：`env_shell/attach_experiment_container.sh`
* 環境変数設定用：`env_shell/set_env.sh`（コンテナ内で実行）

---

## 5. Poetry & 実行方法（コンテナ内）

コンテナに入ったら、以下のようにして環境構築・実行を行う：

```bash
# 1. 依存インストール（初回のみ）
poetry install

# 2. 教師学習
poetry run python -m src.exp.run_teacher

# 3. 生徒ベースライン学習
poetry run python -m src.exp.run_student_baseline

# 4. 蒸留学習
poetry run python -m src.exp.run_distill

# 5. 評価
poetry run python -m src.exp.run_eval_all
```

`cmd/` 配下には、これらをまとめて呼ぶ `.sh` を用意してもよい：

* `cmd/run_teacher_train.sh`
* `cmd/run_student_baseline.sh`
* `cmd/run_distill_dllm2rec.sh`
* `cmd/run_all_pipeline.sh` など

---

## 6. Hydra コンフィグ（概要）

### 6.1 構成

```text
conf/
├── config.yaml        # メイン defaults
├── dataset/
│   ├── movielens.yaml
│   └── amazon_games.yaml
├── teacher/
│   └── ilora.yaml
├── student/
│   ├── sasrec.yaml
│   └── gru4rec.yaml
├── distill/
│   └── dllm2rec.yaml
├── active/
│   ├── all.yaml
│   ├── random.yaml
│   └── meta.yaml
├── paths/
│   └── default.yaml
├── logging/
│   └── default.yaml
├── metrics/
│   └── recsys.yaml
└── hydra/
    └── default.yaml
```

### 6.2 `conf/hydra/default.yaml`

```yaml
hydra:
  run:
    dir: "result/result_${now:%Y%m%d_%H%M%S}"
  job:
    chdir: true
```

→ 各実行結果は `result/result_YYYYMMDD_HHMMSS/` 配下に保存される。

---

## 7. `src/` ディレクトリ構成と役割

```text
src
├── core/
│   ├── paths.py              # data_dir, teacher_outputs_dir 等
│   ├── logging.py            # ログ / TensorBoard / 時間計測
│   ├── metrics.py            # Recall@K, NDCG@K, HitRatio@K
│   ├── data_utils.py         # 前処理 & train/valid/test split
│   ├── seed.py               # 乱数 seed 固定
│   └── git_info.py           # git commit 情報保存
│
├── teacher/
│   ├── interfaces.py         # ITeacherRecommender 抽象
│   ├── ilora_model.py        # ★ iLoRA ロジック再現モデル
│   └── factory.py            # cfg.teacher.name に応じて backend を返す
│
├── student/
│   ├── models.py             # SASRec 等のモデル定義
│   ├── datamodule.py         # DataLoader 構築
│   ├── trainer_baseline.py   # 蒸留なし学習ループ
│   └── evaluator.py          # 評価ロジック
│
├── distill/
│   ├── data_bridge.py        # 教師出力 → 蒸留用テンソル変換
│   ├── kd_losses.py          # DLLM2Rec ロジックの KD loss
│   ├── selection_policy.py   # Active Learning 用インターフェース
│   └── trainer_distill.py    # 蒸留学習ループ
│
└── exp/
    ├── run_teacher.py
    ├── run_student_baseline.py
    ├── run_distill.py
    └── run_eval_all.py
```

---

## 8. iLoRA ロジック再現（teacher/ilora_model.py の要件）

※ 詳細な数学は省略し、構造・処理フローだけ指定します。

1. **ベース LLM**

   * Transformers から LLM を読み込む（例：Llama-2 系）。
   * ベースの重みは基本 freeze。

2. **LoRA エキスパート**

   * 特定の層（Attention / FFN）に LoRA アダプタを挿入。
   * エキスパート数 E（例：4）。
   * エキスパートごとに異なる LoRA パラメータを持たせる。

3. **シーケンス表現**

   * ユーザ履歴シーケンス → テキストプロンプトに変換（形式は iLoRA 実装を参考に）。
   * LLM の出力から、シーケンスレベルの表現ベクトル `h_seq` を取り出す（[CLS]、最後のトークンなど）。

4. **ゲーティングネットワーク**

   * `h_seq` → 全結合層 → softmax → `g_e`（e=1..E）。
   * `g_e` は各エキスパートの重み。シーケンスごとに異なる。

5. **LoRA 更新の合成**

   * 各エキスパートの LoRA 更新 `ΔW_e` を `ΔW = Σ_e g_e * ΔW_e` で合成し、
     LLM の対象層に適用する。
   * これにより、**インスタンス（シーケンス）ごとに異なる有効 LoRA 更新**が使われる。

6. **学習タスク**

   * 入力シーケンス `seq=[i1,..,iT]` に対し、次アイテム `i_{T+1}` を予測する分類タスク。
   * 損失は CE（クロスエントロピー）。
   * 学習対象パラメータ：

     * LoRA パラメータ（全エキスパート分）
     * ゲーティングネットワーク

7. **教師出力の書き出し（蒸留用）**

   * DLLM2Rec 互換形式で以下を出力：

     * `all_embeddings.pt`: `[num_items, d_teacher]` の Tensor（アイテム埋め込み）
     * `myrank_train.txt`: 各トレインシーケンスに対する teacher の top-K 推薦 item ID 列
     * `confidence_train.txt`: 上記に対応するスコア（確率 or ログ確率）
   * 形式は DLLM2Rec を参考にしつつ、自前で定義・実装する。

---

## 9. DLLM2Rec ロジック再現（distill/）

1. **data_bridge.py**

   * iLoRA 教師出力を読み込み、train バッチごとに教師ランキング・スコア・埋め込みを取り出せるようにする。

2. **kd_losses.py**

   * DLLM2Rec の考え方に従い：

     * ランキング蒸留 loss（教師ランキングに合わせる）
     * 埋め込み蒸留 loss（教師埋め込みに近づける）
   * それぞれ cfg.distill のパラメータで重み付けする。

3. **selection_policy.py**

   * `SelectionPolicy` 抽象クラスと `SelectionPolicyAll` 実装。
   * 今は「全サンプルに蒸留をかける」だけでよいが、
     インターフェース設計だけは Random / Meta を追加しやすい形にする。

4. **trainer_distill.py**

   * 生徒モデルに対して：

     * CE loss（通常の next-item prediction）
     * KD loss（ranking + embedding）
   * の和で学習するループを実装。
   * 学習時間は `metrics/time.json` に記録。

---

## 10. 実装メモ：`discussions/docs/specification.md`（必須）

### 10.1 目的

* 実装が途中で止まった場合でも、**次の実装者がここを読むだけで状況を理解できる**ようにするためのメモ。
* 形式は自由だが、**日本語**で書くこと。

### 10.2 テンプレ例

```markdown
# ANYO LLM→Rec 蒸留基盤 実装メモ

## 1. 現在までの実装状況
- core/ : ○○まで実装済み
- teacher/ilora_model.py : ゲーティングまで実装、LoRA合成は未実装 など
- student/models.py : SASRec 完成 / テスト済み
- distill/kd_losses.py : ランキング蒸留のみ実装、embedding蒸留はTODO など

## 2. コンテナ・環境
- 使用Pythonバージョン: 3.x
- 使用コンテナイメージ: anyo-ilora-dllm2rec:latest（Dockerfileからビルド）
- コンテナ起動: `env_shell/start_experiment_container.sh`

## 3. 実行方法のメモ
- 教師学習: `poetry run python -m src.exp.run_teacher dataset=movielens`
- ベースライン: `poetry run python -m src.exp.run_student_baseline ...`
- 蒸留: `poetry run python -m src.exp.run_distill ...`

## 4. 既知の問題 / TODO
- 例: 「Movielens 以外は未検証」
- 例: 「iLoRA のプロンプト設計は暫定版、要改善」

## 5. 次の実装者へのコメント
- 例: 「まず run_teacher を動かして teacher_outputs の形式を確認してください」
- 例: 「DLLM2Rec の data_reader.py を読むと data_bridge のイメージがわきます」
```

---

以上が、**コンテナ設計込みの最終仕様書**です。
このドキュメントとリポジトリの骨組みさえあれば、別の実装者 / エージェントが途中からでも開発を引き継げるようになっています。
