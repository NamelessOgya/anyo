# 生徒ベースライン実行レポート - 2025-11-20 13:07:21

本レポートは、以下の設定で実行された生徒ベースライン学習の結果をまとめたものです。

## 設定

*   **実行日時**: 2025-11-20 13:07:21
*   **エポック数 (max_epochs)**: 20
*   **バッチサイズ (batch_size)**: 1024
*   **データセット**: 全MovieLensデータセット (limit_data_rows: -1)
*   **出力ディレクトリ**: `/content/drive/MyDrive/rec/anyo/result/student_baseline_20251120_130721`

## テストメトリクス

テストセットにおけるモデルのパフォーマンスは以下の通りです。

*   **test_hit_ratio@10**: 0.1392
*   **test_loss**: 6.5213
*   **test_ndcg@10**: 0.0715
*   **test_recall@10**: 0.1392

## チェックポイントの場所

検証時のリコールに基づいて最も性能が良かったモデルのチェックポイントは以下の場所に保存されています。

`/content/drive/MyDrive/rec/anyo/result/student_baseline_20251120_130721/checkpoints/student-baseline-epoch=18-val_recall@10=0.1579.ckpt`
