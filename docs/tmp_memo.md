# Colab環境での教師モデル学習手順 (データ生成・生徒学習スキップ)

このドキュメントは、Google Colab環境で教師モデル（iLoRA）の学習を実行するための手順をまとめたものです。データセットの準備と生徒モデルの学習はスキップします。

## 1. 環境構築

1.  **Poetryのインストール**:
    Poetryがインストールされていない場合、以下のコマンドを実行します。
    ```bash
    bash cmd/colab/00_install_poetry.sh
    ```

2.  **プロジェクト依存関係のインストール**:
    プロジェクトに必要な依存関係をインストールします。
    ```bash
    bash cmd/colab/01_install_dependencies.sh
    ```
    **注意**: `SentencePiece`がないというエラーが発生した場合、`poetry add sentencepiece`を実行し、その後再度`01_install_dependencies.sh`を実行してください。

## 2. Hugging Face認証

LLMモデルにアクセスするためにHugging Faceの認証が必要です。
以下のコマンドを実行し、指示に従ってHugging Faceトークンを入力してください。

```bash
python authenticate_hf.py
```

## 3. 教師モデル学習の実行

事前に学習済みの生徒モデル（SASRec）のチェックポイントを指定して、教師モデルの学習を実行します。

1.  **生徒モデルチェックポイントの指定**:
    以下のパスを、学習済みの生徒モデルチェックポイントのパスとして使用します。
    `/result/student_baseline_20251120_130721/checkpoints/student-baseline-epoch=18-val_recall@10=0.1579.ckpt`
    **注意**: このパスは、環境に応じて調整する必要がある場合があります。

2.  **教師モデル学習の実行**:
    以下のコマンドを実行し、`STUDENT_CHECKPOINT_PATH`には上記で指定したパスを渡します。

    ```bash
    bash cmd/colab/11_run_teacher.sh "/content/drive/MyDrive/rec/anyo/result/student_baseline_20251120_130721/checkpoints/student-baseline-epoch=18-val_recall@10=0.1579.ckpt"
    ```