# プロジェクト仕様書

**LLM → シーケンシャル推薦モデル蒸留基盤（iLoRA + DLLM2Rec, Hydra + Poetry）**

---

## 1. プロジェクト概要

### 1.1 目的

大規模言語モデル（LLM）を用いた推薦モデルから、軽量なシーケンシャル推薦モデルへ知識蒸留を行う実験基盤を構築する。

* **教師モデル**

  * LLM + iLoRA（MoE 型 LoRA 拡張）で学習されたシーケンシャルレコメンダ
  * 1枚の GPU（A100 想定）で学習・推論可能であること

* **生徒モデル**

  * SASRec などの軽量シーケンシャル推薦モデル（ID ベース）

* **蒸留手法**

  * GitHub の **DLLM2Rec** 実装をベースとした蒸留（ランキング＋埋め込み蒸留）

* **管理・運用**

  * 依存関係：Poetry（`pyproject.toml` / `poetry.lock`）
  * 設定管理：Hydra（`conf/` 以下にコンフィグ一式）
  * 実験結果管理：Hydra の `run.dir` を `result/result_{timestamp}` に設定し、その配下にログ・モデル・メトリクス・設定スナップショットを保存

* **将来拡張**

  * Active Learning / Meta Learning による「どのサンプルを蒸留に使うか」の選択を差し込めるように、`SelectionPolicy` インターフェースを用意しておく（現時点では「全サンプル蒸留」のみ実装）

---

## 2. 技術スタックと前提

### 2.1 言語・ランタイム

* Python: 3.11 系（`pyproject.toml` で固定）

### 2.2 主要ライブラリ

* PyTorch
* Transformers
* Hydra (v1.3 系)
* TensorBoard
* その他、iLoRA / DLLM2Rec が依存する標準的な Python ライブラリ（numpy / pandas など）

### 2.3 外部リポジトリ（Git submodule として扱う）

* **iLoRA**（教師モデル用）

  * リポジトリ URL: `https://github.com/AkaliKong/iLoRA.git`
  * 役割: LLM + iLoRA による推薦モデルの学習と推論

* **DLLM2Rec**（蒸留ロジック）

  * リポジトリ URL: `https://github.com/istarryn/DLLM2Rec`
  * 役割: LLM からシーケンシャルモデルへの蒸留（ranking distillation + embedding distillation）

これらは `src/third_party/ilora` / `src/third_party/dllm2rec` として配置し、**原則中身は直接編集しない**。
必要な処理は薄いラッパ層で提供する。

### 2.4 env_shell

* 既に存在する `env_shell` ディレクトリがあり、CUDA / PATH / 環境変数などを設定するシェルスクリプト群がある前提。
* `cmd/*.sh` の中で `source env_shell/xxx.sh` を呼び、その上で `poetry run` で Python を実行する。

---

## 3. ルートディレクトリ構成

```text
.
├── pyproject.toml           # Poetry 定義
├── poetry.lock              # 依存バージョンロック（必ずコミットする）
├── src/                     # すべての Python ソース
├── conf/                    # Hydra コンフィグ
├── cmd/                     # 実験用 .sh スクリプト
├── env_shell/               # 既存の環境設定スクリプト（編集不要）
├── data/                    # 生データ・前処理データ・教師出力
├── result/                  # Hydra の run.dir をこの直下に切る
└── README.md（任意）
```

---

## 4. Poetry 設定

### 4.1 `pyproject.toml`（概要）

実装者は以下のような構成で `pyproject.toml` を作成すること。

```toml
[tool.poetry]
name = "anyo-ilora-dllm2rec"
version = "0.1.0"
description = "LLM->Sequential Rec distillation framework (iLoRA + DLLM2Rec)"
authors = ["<Your Name> <you@example.com>"]
package-mode = false

[tool.poetry.dependencies]
python = "~3.11"
torch = "==2.3.0"
transformers = "==4.43.0"
hydra-core = "^1.3"
numpy = "^1.26"
pandas = "^2.2"
tensorboard = "^2.18"
# 必要に応じて iLoRA / DLLM2Rec が要求する依存ライブラリを追加

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

### 4.2 実行例

```bash
# 依存インストール
poetry install

# 教師学習
poetry run python -m src.exp.run_teacher

# 生徒ベースライン学習
poetry run python -m src.exp.run_student_baseline

# 蒸留学習
poetry run python -m src.exp.run_distill

# 評価
poetry run python -m src.exp.run_eval_all
```

`cmd/*.sh` 内でも、Python 実行は必ず `poetry run` 経由で行うこと。

---

## 5. Hydra コンフィグ仕様

### 5.1 `conf/` 構成

```text
conf/
├── config.yaml        # メイン（defaults の定義）
├── dataset/
│   ├── movielens.yaml
│   └── amazon_games.yaml  # 必要に応じて追加
├── teacher/
│   └── ilora.yaml
├── student/
│   ├── sasrec.yaml
│   └── gru4rec.yaml      # 将来追加用
├── distill/
│   └── dllm2rec.yaml
├── active/
│   ├── all.yaml          # 全サンプル蒸留（現状デフォルト）
│   ├── random.yaml       # 将来追加
│   └── meta.yaml         # 将来追加（Meta/AL 用）
├── paths/
│   └── default.yaml
├── logging/
│   └── default.yaml
├── metrics/
│   └── recsys.yaml
└── hydra/
    └── default.yaml      # hydra.run.dir 等
```

### 5.2 `conf/config.yaml`

```yaml
defaults:
  - dataset: movielens
  - teacher: ilora
  - student: sasrec
  - distill: dllm2rec
  - active: all
  - paths: default
  - logging: default
  - metrics: recsys
  - hydra: default

general:
  experiment_name: "anyo_ilora_dllm2rec"
  seed: 42
  device: "cuda:0"
```

### 5.3 その他のサンプルコンフィグ

#### dataset/movielens.yaml（例）

```yaml
name: "movielens"
max_seq_len: 50
min_user_inter: 5
split:
  method: "leave_one_out"
```

#### teacher/ilora.yaml

```yaml
name: "ilora"
llm_path: "/path/to/Llama-2-7b-hf"
batch_size: 64
lr: 1e-4
num_epochs: 5
prompt_template: "default"
save_interval: 1
```

#### student/sasrec.yaml

```yaml
name: "sasrec"
emb_size: 64
hidden_size: 64
max_seq_len: 50
dropout: 0.1
batch_size: 256
lr: 1e-3
num_epochs: 200
```

#### distill/dllm2rec.yaml

```yaml
name: "dllm2rec"
alpha: 0.5       # ranking distill の重み
ed_weight: 0.3   # embedding distill の重み
lambda: 0.7
batch_size: 256
lr: 1e-3
num_epochs: 200
```

#### active/all.yaml（現状実装対象）

```yaml
name: "all"
strategy: "all"
budget_ratio: 1.0  # Active Learning 実装時に利用予定
```

#### paths/default.yaml

```yaml
data_dir: "data"
result_root: "result"
teacher_outputs_dir: "${paths.data_dir}/teacher_outputs/${dataset.name}"
```

#### hydra/default.yaml

```yaml
hydra:
  run:
    dir: "result/result_${now:%Y%m%d_%H%M%S}"
  job:
    chdir: true
```

> これにより、全ての実行は `result/result_YYYYMMDD_HHMMSS/` 配下で行われる。
> `.hydra/config.yaml` などもそこで保存される。

---

## 6. `src/` ディレクトリ仕様

```text
src
├── core/
│   ├── paths.py              # data_dir, teacher_outputs_dir などパスヘルパ
│   ├── logging.py            # ログ出力 / TensorBoard writer / 時間計測
│   ├── metrics.py            # Recall@K, NDCG@K, HitRatio@K など
│   ├── data_utils.py         # データ前処理・split
│   ├── seed.py               # 乱数 seed 固定
│   └── git_info.py           # git commit ハッシュ取得 & 保存
│
├── teacher/
│   ├── interfaces.py         # ITeacherRecommender, ITeacherExporter 抽象クラス
│   ├── ilora_backend.py      # iLoRA を教師として利用する実装
│   └── factory.py            # cfg.teacher.name に応じた backend を返す
│
├── student/
│   ├── models.py             # SASRec 他のモデル定義
│   ├── datamodule.py         # DataLoader 構築
│   ├── trainer_baseline.py   # 蒸留なし学習ループ
│   └── evaluator.py          # 共通評価ロジック
│
├── distill/
│   ├── data_bridge.py        # teacher_outputs -> テンソル/データセット形式変換
│   ├── kd_losses.py          # DLLM2Rec の ranking / embedding 蒸留損失
│   ├── selection_policy.py   # Active Learning 用インターフェース & 実装
│   ├── trainer_distill.py    # 蒸留学習ループ
│   └── dllm2rec_wrapper.py   # 必要なら DLLM2Rec 公式コードをラップ
│
├── third_party/
│   ├── dllm2rec/             # DLLM2Rec submodule
│   └── ilora/                # iLoRA submodule
│
└── exp/
    ├── run_teacher.py        # 教師モデル学習＋教師出力生成
    ├── run_student_baseline.py
    ├── run_distill.py
    └── run_eval_all.py
```

以下、重要な役割だけ簡潔に説明します。

### 6.1 core

* **paths.py**

  * Hydra の cfg から `data_dir`, `result_root`, `teacher_outputs_dir` を計算する関数を提供。
* **logging.py**

  * Python logger 初期化
  * TensorBoard SummaryWriter 取得（`tb/teacher`, `tb/baseline`, `tb/distill` など）
  * `time_block(name: str)` のような context manager を提供し、処理時間を計測して JSON へ保存。
* **metrics.py**

  * 推薦タスク用の指標を提供：

    * Recall@K
    * NDCG@K
    * HitRatio@K
* **data_utils.py**

  * dataset config に基づき、raw data → processed data を生成し、train/valid/test に分割。
* **seed.py**

  * `set_seed(seed: int)` で Python / numpy / PyTorch の乱数 seed を固定。
* **git_info.py**

  * 現在の `git rev-parse HEAD` を取得し、`git_info.txt` に保存。

### 6.2 teacher

* **interfaces.py**

  * `ITeacherRecommender` と `ITeacherExporter` を定義。例：

    ```python
    class ITeacherRecommender(ABC):
        @abstractmethod
        def train(self, cfg, logger, tb_writer): ...

        @abstractmethod
        def export_for_dllm2rec(self, cfg, logger): ...
    ```

* **ilora_backend.py**

  * 上記インターフェースを実装。
  * `src/third_party/ilora` を import / subprocess 呼び出しなどで利用し、以下を行う：

    1. 教師 LLM + iLoRA モデルの学習
    2. 全アイテムの埋め込み `all_embeddings.pt` の出力
    3. 各訓練シーケンスに対する教師ランキング `myrank_train.txt`
    4. 各ランキングに対する confidence スコア `confidence_train.txt`

* **factory.py**

  * `cfg.teacher.name` に応じて適切な backend インスタンスを生成（現状 `ilora` のみ）。

### 6.3 student

* **models.py**

  * DLLM2Rec が想定する SASRec 実装をベースに、PyTorch モデルとして実装。
* **datamodule.py**

  * dataset config をもとに PyTorch DataLoader を構築。
* **trainer_baseline.py**

  * 生徒モデルを「蒸留なし」で学習するループを実装。
  * CE Loss + optimizer（例：Adam）で学習。
  * validation で Recall@K / NDCG@K を記録。
* **evaluator.py**

  * 学習済みモデルに対して test セットでの指標を計算し、`metrics/eval_*.json` に保存。

### 6.4 distill

* **data_bridge.py**

  * `teacher_outputs_dir` 配下から

    * `all_embeddings.pt`
    * `myrank_train.txt`
    * `confidence_train.txt`
      を読み取り、蒸留用データ（テンソルやデータセット）に変換。

* **kd_losses.py**

  * DLLM2Rec 論文に基づき、以下を実装：

    * 重要度付きランキング蒸留 loss
    * 協調埋め込み蒸留 loss

* **selection_policy.py**

  * Active Learning 用インターフェースと実装を提供。
  * 現時点では以下だけ必須：

    * `SelectionPolicyAll`：全サンプルに蒸留 loss を適用
  * 形だけでよいので、将来 `SelectionPolicyRandom`, `SelectionPolicyMeta` を増やせるような構造にすること。

* **trainer_distill.py**

  * 生徒モデルに DLLM2Rec の蒸留損失を適用して学習するループを実装。
  * イメージ：

    ```python
    policy = build_selection_policy(cfg.active)

    for batch in train_loader:
        student_info = ... # loss, entropyなど必要に応じて
        teacher_info = ... # data_bridgeからの教師関連情報
        mask = policy.select(batch, teacher_info, student_info)

        # mask が True のサンプルについてのみ KD loss 計算
        kd_loss = kd_losses.compute(..., mask=mask)
        ce_loss = ...

        loss = ce_loss + λ * kd_loss
        loss.backward()
        optimizer.step()
    ```

* **dllm2rec_wrapper.py**

  * 必要なら DLLM2Rec 公式コードをラップして使うための窓口。
  * ただし、可能であれば自前の `trainer_distill.py` ＋ `kd_losses.py` で DLLM2Rec と等価の処理を行う。

---

## 7. 実験シナリオと評価要件

### 7.1 実装すべき実験パターン

1. **生徒ベースライン（蒸留なし）**

   * SASRec を CE Loss のみで学習。
   * 評価指標:

     * Recall@K / NDCG@K / HitRatio@K

2. **蒸留あり（DLLM2Rec）**

   * 上記ベースラインを初期値として、DLLM2Rec 蒸留を行うか、
     もしくは蒸留専用の学習として再学習（仕様としてはどちらでもよいが、コード内で選べるようにすると尚良い）。
   * 評価指標:

     * 同上（Recall@K / NDCG@K / HitRatio@K）

3. **将来：蒸留 + Active Learning**

   * 今回は仕様としてフックのみ用意。
   * 実装時は `active=all` のみサポートすればよい。

### 7.2 時間計測

以下の時間を計測し、`result/result_{timestamp}/metrics/time.json` に保存。

* `teacher_train_time_total`
  教師モデルの学習時間

* `distill_teacher_export_time`
  教師モデルから DLLM2Rec 用ファイルを生成する時間

* `student_baseline_train_time`
  生徒ベースライン学習時間

* `distill_student_train_time`
  DLLM2Rec 蒸留学習時間

### 7.3 ログ・メトリクス保存

* Hydra によって run.dir = `result/result_{timestamp}` が自動生成される。
* その配下に以下を保存：

```text
result/result_{timestamp}/
  ├── .hydra/                  # Hydra の config スナップショット等
  ├── logs/
  │   ├── train_teacher.log
  │   ├── train_student_baseline.log
  │   └── train_distill.log
  ├── tb/
  │   ├── teacher/
  │   ├── baseline/
  │   └── distill/
  ├── models/
  │   ├── teacher/
  │   ├── baseline/
  │   └── distill/
  └── metrics/
      ├── eval_teacher.json（教師評価をする場合）
      ├── eval_baseline.json
      ├── eval_distill.json
      └── time.json
```

---

## 8. cmd スクリプトの仕様（概要）

`cmd/` 配下には、以下のようなシェルスクリプトを用意する。

* `run_teacher_train.sh`

  * `source env_shell/...`
  * `poetry run python -m src.exp.run_teacher dataset=... teacher=ilora`

* `run_student_baseline.sh`

  * 同様に `poetry run python -m src.exp.run_student_baseline ...`

* `run_distill_dllm2rec.sh`

  * `poetry run python -m src.exp.run_distill ...`

* `run_eval_all.sh`

  * `poetry run python -m src.exp.run_eval_all ...`

* `run_all_pipeline.sh`

  * 上記を順に呼び出して、一括で教師→生徒→蒸留→評価を実行する。
