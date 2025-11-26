# Flash Attention 2 の Poetry 環境へのインストール方法に関する調査

## 1. 概要
`flash-attn` はPyTorchの拡張であり、CUDA環境への依存性が高いため、Poetryプロジェクトへの導入にはいくつかの考慮事項がある。特に、PyTorchのCUDAバージョンとシステムのCUDAバージョンとの互換性、およびビルド時の依存関係解決が重要となる。

## 2. インストール手順とベストプラクティス

### 2.1. CUDA環境の確認
`flash-attn`のインストール前に、システムに互換性のあるCUDAツールキットがインストールされていることを確認する。FlashAttention-2では、一般的にCUDA 12.0以上が推奨される。PyTorchがどのCUDAバージョンでビルドされたかを確認し、システムと一致させる必要がある。

### 2.2. PyTorchのインストール（CUDAサポート付き）
`flash-attn`はPyTorchの拡張であるため、CUDAサポート付きのPyTorchが必須となる。
`pyproject.toml`に`torch`を追加する際は、適切なCUDAプラットフォームを指定する。
例: `poetry add "torch==2.9.1+cu121" --source torch --group dev`

### 2.3. `flash-attn`の追加（推奨される順序）

#### 2.3.1. `poetry add` による試行
最もシンプルな方法は、`poetry add`コマンドで直接追加することである。Poetryは、Python、PyTorch、CUDAのバージョンに合致するプリビルドされたホイールパッケージを見つけようと試みる。
```bash
poetry add flash-attn
```
この方法が成功すれば、通常は`nvcc`（CUDA C++コンパイラ）を完全にインストールする必要はなく、CUDAランタイムライブラリのみで動作する。

#### 2.3.2. `pip install --no-build-isolation` による回避策
`poetry add flash-attn` が失敗する場合、`flash-attn`がPEP 517ビルド分離に関する問題に直面することがある（例: ビルドプロセス中に`ModuleNotFoundError: No module named 'torch'`が発生する）。この場合、Poetryの仮想環境内で`pip`を使用してビルド分離を無効にしてインストールするのが一般的な回避策である。ビルドを高速化するため、`ninja`を先にインストールしておくことが推奨される。
```bash
poetry run pip install ninja
poetry run pip install flash-attn --no-build-isolation
```
このコマンドは、`pip`に対し隔離されたビルド環境を作成せずに`flash-attn`をインストールするように指示し、Poetryの仮想環境内の`torch`インストールにアクセスできるようにする。

#### 2.3.3. Gitリポジトリからのインストール（最終手段）
プリビルドされたホイールに問題が続く場合や、特定の開発バージョンが必要な場合は、GitHubリポジトリから直接インストールすることも可能である。この方法はソースからのビルドを伴うため、システムに`nvcc`（CUDA >= 11.7）とC++17コンパイラが利用可能である必要がある。
```bash
poetry run pip install git+https://github.com/Dao-AILab/flash-attention.git
```

## 3. 環境変数（詳細な制御用）
必要に応じて、以下の環境変数を使用することで、ビルドプロセスをより詳細に制御できる。
*   `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`: 明示的にコンパイルを防ぎ、プリビルドされたホイールのみを使用する場合に設定する。
*   `CUDA_HOME`: ビルドプロセス中のCUDA検出を助けるため。
*   `FLASH_ATTENTION_FORCE_BUILD=TRUE`: プリビルドされたホイールが利用可能であっても、強制的にソースからコンパイルする場合。

## 4. 結論
Poetry環境での`flash-attn`のインストールは、まず`poetry add flash-attn`を試し、失敗した場合は`poetry run pip install flash-attn --no-build-isolation`を使用することが最も堅牢な方法である。

---
**参考資料:**
- Hugging Face Transformersのドキュメント
- `flash-attn` GitHubリポジトリ
- 各種コミュニティフォーラム (PyPI, Reddit, Mediumなど)
