# Colab環境向け実装計画書

## 1. 目的

本プロジェクトを、DockerコンテナをビルドできないGoogle Colab（または同様の非コンテナ環境）で実行可能にすることを目的とする。
その際、既存のDockerベースの実行フローは完全に維持し、両方の環境での実行をサポートする。

## 2. 基本方針

-   [ ] **既存機能の維持**: `Dockerfile`, `cmd/`, `env_shell/` 配下など、既存のDocker関連ファイルやスクリプトは変更しない。
-   [ ] **Colab用ディレクトリの作成**: Colab環境専用の実行スクリプトを格納するため、新たに `cmd/colab/` ディレクトリを作成する。これにより、既存の実行体系との分離を図る。
-   [ ] **依存関係の管理**: Colab環境においても `pyproject.toml` と `poetry.lock` に基づいた依存関係を再現するため、Poetryを利用する。環境に直接Poetryをインストールし、`poetry install` コマンドでパッケージを導入する。
-   [ ] **段階的な実行**: 環境構築、データ準備、各モデルの学習・評価をステップごとに実行できるスクリプトを準備し、Colab環境でのデバッグと実行を容易にする。

## 3. 作業計画

### ステップ1: 環境構築

Colabセッション内で、プロジェクトの依存関係をインストールするための基盤を整える。

-   **タスク1.1: Poetryのインストール**
    -   [x] Colab環境に`pip`を用いてPoetryをインストールする。
    -   成果物: `cmd/colab/00_install_poetry.sh`

-   **タスク1.2: プロジェクト依存関係のインストール**
    -   [x] Poetryを使い、`pyproject.toml` および `poetry.lock` に基づいて、PyTorchなど必要なライブラリをすべてインストールするスクリプトを作成する。
    -   [x] A100 GPU環境での動作を想定し、CUDAの互換性があるバージョンがインストールされることを`poetry.lock`に期待する。
    -   成果物: `cmd/colab/01_install_dependencies.sh`

### ステップ2: データ準備

実験に必要なデータセットをダウンロード・展開する。

-   **タスク2.1: データセット準備スクリプトの作成**
    -   [x] 既存の `data/download_movielens.sh` を呼び出す、Colab用のラッパースクリプトを作成する。
    -   成果物: `cmd/colab/02_prepare_dataset.sh`

### ステップ3: 実行スクリプトの作成

Colab環境から各実験（生徒ベースライン、教師学習、蒸留）を順次実行するためのスクリプトを作成する。これらのスクリプトは、チェックポイントのパス解決など、手動での設定変更を極力減らす工夫を検討する。

-   **タスク3.1: 生徒モデル（ベースライン）の学習・評価**
    -   [x] `run_student_baseline.py` を実行するためのスクリプトを作成する。
    -   成果物: `cmd/colab/10_run_student_baseline.sh`

-   **タスク3.2: 教師モデル（iLoRA）の学習・評価**
    -   [x] `run_teacher.py` を実行するためのスクリプトを作成する。
    -   [x] このスクリプトは、先行する生徒モデル学習で生成されたチェックポイントのパスを自動的に検出し、`teacher.rec_model_checkpoint_path` として引数で渡す機能を検討する。
    -   成果物: `cmd/colab/11_run_teacher.sh`

-   **タスク3.3: 知識蒸留の実行と評価**
    -   [x] `run_distill.py` を実行するためのスクリプトを作成する。
    -   [x] このスクリプトは、先行する教師モデル学習で生成されたチェックポイントのパスを自動的に検出し、`distill.teacher_checkpoint_path` として引数で渡す機能を検討する。
    -   成果物: `cmd/colab/12_run_distill.sh`

-   **タスク3.4: 全実験の統合実行（オプション）**
    -   [x] 上記 `10_` から `12_` までのスクリプトを順番に実行する統合スクリプトを作成する。これにより、ワンコマンドで全実験を再現可能にする。
    -   成果物: `cmd/colab/run_all_experiments.sh`

### ステップ4: 動作確認

-   [x] 作成したスクリプトをColab環境で順に実行し、エラーなく全実験が完了することを確認する。
-   [ ] 特に、各ステップ間でのチェックポイントの受け渡しが正しく行われることを検証する。

## 4. 成果物

-   [x] `docs/specification/08_implement_for_colab.md` （本ドキュメント）
-   [x] `cmd/colab/` ディレクトリ
    -   [x] `00_install_poetry.sh`
    -   [x] `01_install_dependencies.sh`
    -   [x] `02_prepare_dataset.sh`
    -   [x] `10_run_student_baseline.sh`
    -   [x] `11_run_teacher.sh`
    -   [x] `12_run_distill.sh`
    -   [x] `run_all_experiments.sh` （オプション）

## 5. 進捗報告 (2025-11-20)

当初の計画に沿ってColab環境での実行準備を進めたが、複数の問題に直面した。以下にその内容と解決策をまとめる。

### 5.1. 直面した問題と解決策

1.  **依存関係の解決失敗 (`triton`の非互換性)**
    -   **問題:** `poetry install` を実行したところ、`pyproject.toml` にてバージョンが固定されていた `torch==2.1.2` が要求する `triton==2.1.0` が、ColabのPython環境と互換性がなくインストールに失敗した。
    -   **解決策:** `pyproject.toml` の `torch` や `transformers` などのバージョン指定を、特定バージョンに固定 (`X.Y.Z`) するのではなく、互換性のある新しいバージョンを許容 (`^X.Y.Z`) するように変更。その後、`poetry.lock` を削除し、`poetry install` を再実行することで、現在の環境と互換性のある依存関係を再解決し、インストールに成功した。また、`tensorboard` が不足していたため、これを明示的に依存関係に追加した。

2.  **データの欠損 (`FileNotFoundError`)**
    -   **問題:** データローダー (`SASRecDataModule`) が `train.csv`, `val.csv`, `test.csv` を要求するが、これらのファイルがプロジェクト内に存在しなかった。
    -   **解決策:** `src/core/preprocess_data.py` に、`ratings.dat` から上記の3ファイルを生成する前処理ロジックを発見。データ準備スクリプト (`cmd/colab/02_prepare_dataset.sh`) を更新し、元データをダウンロードした後にこの前処理スクリプトを実行するように修正した。

3.  **謎のサイレント終了バグ**
    -   **問題:** `10_run_student_baseline.sh` を実行しても、エラーメッセージなし・終了コード0でプロセスが完了するにもかかわらず、チェックポイントやログが一切生成されない、最も解決困難な問題に直面した。
    -   **調査:** 詳細なログ出力 (`python -v`) や、単純なテストスクリプト (`test_hydra.py`) の実行により、問題がHydraの `@hydra.main` デコレータと `pytorch_lightning` の相互作用に起因する可能性を特定した。デコレータが原因で、スクリプト本体のロジックが実行される前に、例外が握りつぶされサイレントに終了していた。
    -   **解決策:** `@hydra.main` デコレータの使用を中止し、スクリプト内で `hydra.initialize()` と `hydra.compose()` を使って手動でHydraを初期化する方式にリファクタリングした。これにより、握りつぶされていた例外が表面化し、デバッグが可能になった。

4.  **連鎖的なバグの修正**
    -   サイレント終了問題を解決したことで、以下のような潜在的なバグが次々と明らかになった。
        -   `dm.prepare_.data()` というタイポ (`AttributeError`)。
        -   手動初期化時に `HydraConfig` が取得できない問題 (`ValueError`)。
        -   `CustomRichProgressBar` コールバック内の複数の `AttributeError` と `TypeError`。
    -   **解決策:** これら全てのバグを特定し、一つずつ修正した。

5.  **Hydra引数パース問題**
    -   **問題:** 蒸留学習スクリプト(`12_run_distill.sh`)実行時に、Hydraの設定オーバーライド（特にパスを含む引数）が正しくパースされず、`Hydra initialization failed: mismatched input '=' expecting <EOF>`エラーが発生した。引数をシングルクォートで囲んだり外したりしても解決しなかった。
    -   **解決策:** 設定ファイル(`conf/distill/dllm2rec.yaml`)を直接修正し、`distill.teacher_checkpoint_path`と`distill.teacher_outputs_batches_dir`の値をYAMLファイル内に直接記述するようにした。これにより、コマンドラインでのパース問題を回避した。

6.  **`run_distill.py`内の`NameError`**
    -   **問題:** `src/exp/run_distill.py`内で`teacher_checkpoint_file`という変数名が`teacher_checkpoint_file_path`と誤って参照されている`NameError`が複数箇所で発生した。
    -   **解決策:** `src/exp/run_distill.py`内の全ての`teacher_checkpoint_file`の誤参照を`teacher_checkpoint_file_path`に修正した。

### 5.2. 現在の状況

-   上記すべての問題を解決し、`cmd/colab/10_run_student_baseline.sh` を実行して、生徒モデル（ベースライン）の学習・評価・チェックポイント保存が正常に完了することを確認した。
-   動作確認のため、`conf/dataset/movielens.yaml` の `limit_data_rows` を `10000` に設定している。
-   次のステップとして、教師モデルの学習 (`11_run_teacher.sh`) に進む準備が整った。

### 5.3. 教師モデル学習の実行とデバイス不整合問題

-   `11_run_teacher.sh` を実行し、教師モデルの学習に着手した。
-   **進捗:** 以前の開発を妨げていた `NaN` / `inf` 損失の問題が、今回の環境およびコードベースでは発生せず、学習と検証が正常に完了した。これは、依存関係の更新やHydra初期化方法のリファクタリングが、間接的に数値不安定性問題を解決したことを示唆する大きな進展である。
-   **新たな問題:** 学習とテストが完了した後、訓練データセットに対する教師出力を生成するフェーズで、`RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu` が発生した。
-   **原因分析:** このエラーは、`iLoRAModel` に含まれる `rec_model` (凍結されたSASRecモデル) が、`pl.Trainer` による学習・評価プロセスが終了した後、CPU上に配置されたままになっていることが原因である。その後、スクリプトが手動でGPU上のバッチデータを処理しようとするため、デバイスの不整合が発生していた。
-   **解決策:** この問題を解決するため、`src/exp/run_teacher.py` を修正。学習完了後に手動でデバイスを管理するのではなく、`pl.Trainer`によって学習・評価された `LightningModule` (`iLoRATrainer`) 全体を、推論ループの直前に `.to(device)` を用いて明示的にGPUへ移動させるように変更した。

### 5.4. 教師モデル実行の成功と知識蒸留への移行

-   **実行成功:** 上記のデバイス不整合問題の修正後、再度 `11_run_teacher.sh` を実行した結果、教師モデルの学習、評価、および教師出力のバッチ保存まで、すべてのプロセスがエラーなく正常に完了した。
-   **次のステップ（知識蒸留）:**
    1.  `src/exp/run_distill.py` を確認したところ、このスクリプトは既にHydraの手動初期化方式にリファクタリング済みであり、さらに最新の教師モデルの成果物（コンフィグ、チェックポイント、教師出力）を動的に発見するロジックも実装されていた。これにより、コンフィグファイルの手動更新が不要になっている。
    2.  `cmd/colab/12_run_distill.sh` スクリプトを、メモリ使用量を考慮してバッチサイズを小さく設定して実行するように修正した。
    3.  次のタスクとして、この `12_run_distill.sh` を実行し、蒸留プロセスの動作確認とデバッグを行う。

### 5.5. 知識蒸留の実行と完了

-   **実行と段階的なエラー修正:** `12_run_distill.sh` を実行する過程で、以下の複数のエラーに直面したが、それぞれを特定し修正した。
    1.  **`TypeError: unhashable type: 'list'`**: 傾向スコア計算時に発生。`run_distill.py`でテンソル形状を修正して解決。
    2.  **`TypeError: DistillationTrainer.__init__() missing ...`**: コンストラクタ引数不足。`run_distill.py`で引数を追加して解決。
    3.  **`RuntimeError: 0D or 1D target tensor expected`**: `CrossEntropyLoss`での次元不整合。`trainer_distill.py`内で`.squeeze(-1)`を適用して解決。
    4.  **`RuntimeError: Index tensor must have the same number of dimensions`**: ネガティブサンプリング処理での次元不整合。`trainer_distill.py`の`training_step`冒頭で`next_item`の形状を正規化することで解決。
    5.  **`MisconfigurationException: No \`test_step()\` method defined`**: `run_distill.py`に最終評価ステップを追加した際に発生。`DistillationTrainer`に`test_step`メソッドが定義されていなかったため、`validation_step`をコピーしログのプレフィックスを`test_`に変更することで解決。

-   **実行成功:** 上記すべてのエラーを解決した後、再度 `12_run_distill.sh` を実行した結果、知識蒸留の学習プロセスおよび最終評価がエラーなく正常に完了した。

### 5.7. 本番環境での実験開始

Colab環境での基本的な動作確認が完了したため、A100 GPUの本番環境にて、全データセットを用いた学習・評価を開始する。

1.  **本番環境(A100)向けコンフィグの整備**
    -   [x] 生徒モデル、教師モデル、蒸留学習のそれぞれに対して、バッチサイズを`512`に増加させた本番用の設定ファイル (`conf/train/*_production.yaml`) を作成した。
    -   [x] 各実行スクリプト (`src/exp/run_*.py`) が、コマンドライン引数で設定をオーバーライドできるように修正した。
    -   [x] `conf/config.yaml`からデフォルトの`train`設定を削除し、各スクリプトで明示的に`train=...`を指定するように変更した。
    -   [x] 全ての学習ステージでバッチサイズが`512`となるように、`conf/train/teacher_production.yaml`と`conf/train/distill_production.yaml`の`batch_size`を`512`に設定し、対応する`cmd/colab/*.sh`スクリプトの`train.batch_size`オーバーライドを削除した。

2.  **本番データでの完全な実験実行と評価**
    -   [x] `conf/dataset/movielens.yaml` の `limit_data_rows` を `-1` に設定し、全データを使用するように変更した。
    -   [ ] 生徒ベースライン、教師モデル、蒸留学習の全パイプラインを本番設定で実行する。
        -   [x] `cmd/colab/10_run_student_baseline.sh train=student_production` (完了)
        -   [x] `cmd/colab/11_run_teacher.sh train=teacher_production` (前提条件として完了、`teacher_outputs_batches_dir` 生成のためデータ1000件で動作確認済み)
        -   [x] `cmd/colab/12_run_distill.sh train=distill_production` (データ1000件で動作確認済み)

-   **次のステップ**: 完全なデータセット (`limit_data_rows: -1`) を使用して、教師モデルおよび蒸留学習を実行します。

### 5.8. チェックポイントの自動検索機能の廃止と明示的なパス指定への変更

-   **問題**: `cmd/colab/11_run_teacher.sh` および `cmd/colab/12_run_distill.sh` は、以前の実行結果から最新のチェックポイントパスを自動的に検索するロジックを含んでいます。しかし、この自動検索が意図しない結果を招いたり、特定のチェックポイントを使用したい場合に不便であることが判明しました。特に、教師学習では学生モデルのチェックポイントが、蒸留学習では教師モデルのチェックポイントが必要ですが、自動検索機能が誤ったパスを選択し、エラーの原因となっていました。

-   **計画変更**: 自動検索機能を廃止し、実行スクリプトがチェックポイントパスを明示的な引数として受け取るように変更します。これにより、使用するチェックポイントを正確に制御できるようになります。

-   **変更内容**:
    1.  `cmd/colab/11_run_teacher.sh` を修正し、学生モデルのチェックポイントパスを引数として受け取るようにしました。スクリプト内の`LATEST_STUDENT_DIR`と`CHECKPOINT_FILE`を自動で検索するロジックは削除され、代わりに引数で渡されたパスを使用するように変更しました。
    2.  `cmd/colab/12_run_distill.sh` は、元々引数を`src.exp.run_distill`に直接渡すシンプルな構造でした。そのため、自動検索ロジックの削除は不要で、教師モデルのチェックポイントパスと教師出力バッチディレクトリのパスは引数として明示的に渡すことで制御します。これらのパスは設定ファイル(`conf/distill/dllm2rec.yaml`)で指定するか、コマンドライン引数でオーバーライドして渡すことになります。

### 5.9. 性能改善
    -   [ ] 上記のメトリクスをベースラインとして、`implement.md`や`02_development_notes_ja.md`で挙げられているハイパーパラメータチューニング（損失重み、プロンプト設計、LLMアクセス頻度削減など）に焦点を当て、蒸留モデルの性能向上に取り組む。
