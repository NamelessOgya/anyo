# Llamaモデル実行設定の最適化計画

## 1. 事前調査フェーズ

- [x] **タスク1.1**: `ref_repositories/iLoRA` ディレクトリ内のファイルを確認し、先行研究で用いられた以下のハイパーパラメータと設定を特定する。
    - [x] `train_movielens.sh` から `batch_size`, `lr` (学習率) を調査する。
        - **結果**: `batch_size = 8`, `lr = 8e-4`
    - [x] 実行スクリプトや `main.py` から `num_workers` の設定を調査する。
        - **結果**: `num_workers = 8` (main.pyのArgumentParserのデフォルト値)
    - [x] `train_movielens.sh` に記載のある `--prompt_path` (`./prompt/movie.txt`) の内容を確認し、使用されているプロンプトを特定する。
        - **結果**: `ref_repositories/iLoRA/prompt/movie.txt` には複数のプロンプトテンプレートが存在。
            - `This user has watched [HistoryHere] in the previous. Please predict the next movie this user will watch. Choose the answer from the following 20 movie titles: [CansHere]. Answer:`
            - `This user has watched [HistoryHere] in the previous. Given the following 20 movie titles: [CansHere], recommend one movie for this user to watch next. The movie title you recommend is:`
            - `The visit history of this user is: [HistoryHere]. Recommend a next movie for this user to watch from the following movie title set: [CansHere]. The recommendation should contain one movie title only. Recommendation:`
- [x] **タスク1.2**: 一般的なLoRA学習において `num_workers` が実行速度に与える影響について調査し、その結果を `docs/research/03_about_num_workers.md` に日本語でまとめる。
    - **結果**: `docs/research/03_about_num_workers.md` を作成し、`num_workers`の役割、最適化のポイント、メモリ影響、`persistent_workers`についてまとめました。

## 2. 最適設定の探索フェーズ (q-LoRA有効)

- [x] **タスク2.1**: 事前調査フェーズの結果とq-LoRAによる軽量化を考慮し、`batch_size=128`および`batch_size=256`をベースラインとして最適な `num_workers` の組み合わせを探索する実験計画を立案する。
    - **目的**: Llamaモデル (q-LoRA有効) の学習において、GPUメモリ使用量を抑えつつ、学習時間を最適化する `num_workers` と `batch_size` の組み合わせを特定する。
    - **評価指標**:
        - **学習時間**: 各エポックの完了までにかかる時間、または全トレーニングの総時間。
        - **GPUメモリ使用量**: `torch.cuda.max_memory_allocated()` を使用してピークメモリ使用量を測定。
    - **探索範囲**: `limit_data_rows: 1000` (高速なイテレーションのため) および `max_epochs: 1` で実験を実施。
        - **実験1: `batch_size=128` の場合**
            - `batch_size: 128`, `num_workers: 0`
            - `batch_size: 128`, `num_workers: 4` (現在のLlama q-LoRA成功設定)
            - `batch_size: 128`, `num_workers: 8` (iLoRA参照値)
            - `batch_size: 128`, `num_workers: 11` (Pytorch Lightningの推奨値)
        - **実験2: `batch_size=256` の場合**
            - [x] `batch_size: 256`, `num_workers: 0`
                - **結果**: (前の実行で`train=teacher`を明示していなかったため、実効バッチサイズは`conf/train/default.yaml`の`64`が適用されていた)
                    - **実効バッチサイズ 64 の場合**:
                        - 学習時間 (1 epoch): ~135秒
                        - 総学習時間 (trainer.fit): ~266秒
                        - ピークGPUメモリ使用量: 11.35 GB
                    - **実効バッチサイズ 256 の場合**: (`train=teacher`を明示的に指定した後の実行)
                        - 学習時間 (1 epoch): ~125秒
                        - 総学習時間 (trainer.fit): ~256秒
                        - ピークGPUメモリ使用量: 32.23 GB
                    - **時間比較**: `batch_size=256` は `batch_size=64` よりも1エポックあたり約10秒高速 (約7%削減) でしたが、GPUメモリ使用量は約3倍に増加しました。
            - [ ] `batch_size: 256`, `num_workers: 4` (in_progress)
            - [ ] `batch_size: 256`, `num_workers: 8`
            - [ ] `batch_size: 256`, `num_workers: 11`
- [ ] **タスク2.2**: タスク2.1で立案した実験計画を実行し、各設定でのパフォーマンスを測定・比較する。
- [ ] **タスク2.3**: 実験結果に基づき、最適な `num_workers` と `batch_size` の組み合わせを決定し、その設定をデフォルトのコンフィグファイルに反映する。
