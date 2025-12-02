# Config構成の精査とリファクタリング提案

## 現状の分析 (Current Analysis)

現在の`conf`ディレクトリおよび`config.yaml`を精査した結果、以下の問題点が確認されました。

### 1. 役割の重複 (Role Duplication)
最も大きな問題は、**モデル定義ファイル (`conf/teacher/*.yaml`) が、学習設定 (`conf/train/*.yaml`) やデータセット設定 (`conf/dataset/*.yaml`) の役割を侵食している**点です。

*   **学習パラメータの重複:**
    *   `conf/teacher/ilora.yaml`, `bigrec.yaml`, `moe_bigrec.yaml` のすべてにおいて、`batch_size`, `learning_rate`, `val_check_interval`, `accumulate_grad_batches`, `warmup_steps` などが定義されています。
    *   これらは本来 `conf/train/teacher.yaml` や `conf/train/default.yaml` で管理されるべき項目です。
    *   現状では、`conf/train` の設定を変更しても `conf/teacher` の設定で上書きされる（あるいはその逆）可能性があり、意図しない設定で学習が走るリスクがあります。

*   **データセット設定の混入:**
    *   `conf/teacher/moe_bigrec.yaml` に `limit_data_rows` が定義されています。これは `conf/dataset` の責務です。

*   **`config.yaml` の存在:**
    *   ルートディレクトリに `config.yaml` が存在し、`conf/` 以下の設定と内容が重複しています（`teacher`, `student`, `train`, `dataset` セクションなど）。
    *   Hydraを使用している場合、`conf/` ディレクトリによる構成管理が主となるため、このモノリシックな `config.yaml` は混乱の元となります。

### 2. 責務の混在 (Mixed Responsibilities)
*   **Teacher Config:** モデルのアーキテクチャ（層の数、隠れ層のサイズなど）だけでなく、学習のハイパーパラメータやデータセットのフィルタリング設定まで含んでしまっています。

---

## 理想的なConfig構成案 (Ideal Configuration Proposal)

「漏れなく被りなく (MECE)」かつ「責務の分離 (Separation of Concerns)」を実現するための構成を提案します。

### 1. 構成の原則
各Configグループの責務を以下のように厳密に定義します。

*   **`model` (旧 `teacher`/`student`):**
    *   **責務:** モデルのアーキテクチャ定義のみ。
    *   **含まれるパラメータ:** `hidden_size`, `num_layers`, `dropout_rate`, `model_type`, `llm_model_name`, `lora_r` など。
    *   **含まないパラメータ:** `batch_size`, `learning_rate`, `epochs`。

*   **`train`:**
    *   **責務:** 学習ループとオプティマイザの設定。
    *   **含まれるパラメータ:** `batch_size`, `max_epochs`, `learning_rate`, `weight_decay`, `optimizer`, `scheduler`, `val_check_interval`, `accumulate_grad_batches`, `accelerator`, `devices`。

*   **`dataset`:**
    *   **責務:** データの場所と前処理の設定。
    *   **含まれるパラメータ:** `data_dir`, `dataset_name`, `limit_data_rows`, `max_seq_len` (データ依存の場合)。

*   **`experiment`:**
    *   **責務:** 上記のコンポーネントを組み合わせる (Composition)。
    *   **役割:** 特定の実験（例: "MovieLensでのiLoRA学習"）に必要な `model`, `train`, `dataset` の組み合わせを指定し、必要であれば特定のパラメータのみをオーバーライドする。

### 2. 提案するディレクトリ構成

```text
conf/
├── dataset/            # データセット設定
│   ├── movielens.yaml
│   └── ...
├── model/              # モデルアーキテクチャ設定 (teacher/studentを統合またはサブディレクトリ化)
│   ├── teacher/
│   │   ├── ilora.yaml
│   │   ├── bigrec.yaml
│   │   └── moe_bigrec.yaml
│   └── student/
│       └── sasrec.yaml
├── train/              # 学習ループ設定
│   ├── default.yaml    # 共通のデフォルト設定
│   ├── pretrain.yaml   # 事前学習用 (例: teacher用)
│   └── distill.yaml    # 蒸留用
├── experiment/         # 実験構成 (Composition Root)
│   ├── ilora_movielens.yaml
│   ├── bigrec_movielens.yaml
│   └── ...
└── hydra/              # Hydra自体の設定 (ログ出力先など)
```

### 3. 具体的なリファクタリング手順

1.  **`conf/teacher/*.yaml` の清掃:**
    *   `batch_size`, `learning_rate`, `val_check_interval`, `limit_data_rows` などの学習・データセット関連パラメータを削除します。
    *   純粋なモデルパラメータのみを残します。

2.  **`conf/train/*.yaml` の整備:**
    *   削除した学習パラメータを `conf/train` 配下の適切なファイル（例: `teacher.yaml` を `pretrain.yaml` にリネームするなどして整理）に集約します。
    *   モデルごとに異なる学習設定が必要な場合は、`conf/train/bigrec_train.yaml` のように作成するか、`experiment` ファイル内でオーバーライドします。

3.  **`config.yaml` の廃止:**
    *   `config.yaml` を削除し、すべての設定を `conf/` 以下のHydra構成に移行します。

4.  **`experiment` ファイルの修正:**
    *   `defaults` リストを更新し、新しいディレクトリ構成 (`model/teacher` など) を反映させます。

この構成により、モデルの構造を変えずに学習率だけ調整したい場合は `train` configを、データセットを変えたい場合は `dataset` configを切り替えるだけで済み、設定の重複や予期せぬ上書きを防ぐことができます。
