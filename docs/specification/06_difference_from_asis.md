# 既存実装と参照リポジリのロジックレベルの差分 (Logic-Level Difference from As-Is)

このドキュメントは、本プロジェクトの `src` ディレクトリ内の実装が、参照リポジトリ（iLoRA, DLLM2Rec）で提案されている**既存手法のロジックを再現できているか**に焦点を当て、その差分をまとめるための依頼文書です。アーキテクチャや実装の詳細な構造が異なっていても構いません。重要なのは、**アルゴリズムの主要なステップや数式が、本プロジェクトのコードでどのように実装され、既存ロジックの再現がなされているか**という点です。

---

## 1. 比較方針

*   **目的**: 本プロジェクトのコードが、`00_overview.md` に記載されている iLoRA および DLLM2Rec の「ロジック再現の方針」をどの程度満たしているかを客観的に評価する。
*   **粒度**: 各アルゴリズムの核となるロジック、数式、データフローの主要な変換ステップに焦点を当てる。コードの行単位の比較ではなく、アルゴリズムの再現性に着目する。
*   **対象**:
    *   **iLoRA ロジック**: `ref_repositories/iLoRA/` の主要なロジックと `src/teacher/` の対応する実装。
    *   **DLLM2Rec ロジック**: `ref_repositories/DLLM2Rec/` の主要なロジックと `src/distill/` の対応する実装。

---

## 2. iLoRA ロジックの再現性評価 (`src/teacher/` vs `ref_repositories/iLoRA/`)

### 2.1. iLoRA の主要ロジック (00_overview.md より)

*   **LLM + 複数 LoRA エキスパート + シーケンスごとのゲーティング。**
*   **`h_seq`（シーケンス表現） → softmaxゲート → LoRAパラメータを線形結合 → LLMに適用。**
*   **学習：次アイテム予測（CE loss）、学習対象は LoRA + ゲートのみ。**
*   **教師出力として DLLM2Rec 互換のランキング・スコア・埋め込みを出す。**

### 2.2. 評価項目と確認事項

以下の点について、本プロジェクトの `src/teacher/ilora_model.py` および関連ファイルが、参照実装のロジックを再現しているかを確認してください。

*   **ゲーティングネットワークの機能**:
    *   シーケンス表現 (`h_seq` に相当するもの) がどのように生成され、ゲーティングネットワークに入力されているか。
    *   ゲーティングネットワークが、複数の LoRA エキスパートに対する softmax 重みを正しく出力しているか。
*   **LoRA パラメータの線形結合**:
    *   ゲーティングネットワークの重みに基づいて、複数の LoRA エキスパートの出力（またはパラメータ自体）がどのように線形結合され、LLM に適用されているか。
    *   特に、`peft` ライブラリの利用方法が、参照実装の意図する動的な LoRA 結合ロジックを再現しているか。
*   **LLM への適用と次アイテム予測**:
    *   結合された LoRA が適用された LLM が、どのように次アイテムの予測（ロジット）を生成しているか。
    *   プロンプトエンジニアリングが、LLM の入力として適切に機能しているか。
*   **学習対象**:
    *   学習対象が LoRA パラメータとゲーティングネットワークのパラメータのみに限定されているか（LLM の基盤モデルはフリーズされているか）。
    *   損失関数が次アイテム予測の Cross-Entropy Loss を使用しているか。
*   **教師出力**:
    *   `get_teacher_outputs` メソッドが、DLLM2Rec 互換のランキングスコアと埋め込みを正しく出力しているか。
    *   特に、埋め込みが LLM の最終隠れ状態を適切に集約したものであるか。

### 2.3. 評価結果

#### 項目: ゲーティングネットワークの機能
*   **参照実装のロジック概要**: `ref_repositories/iLoRA/model/model_interface.py` の `MInterface` クラスの `forward` メソッドで `self.router(user_embeds)` を呼び出し、推薦モデルから得られたユーザー埋め込み (`user_embeds`) をゲーティングネットワークに入力。ゲーティングネットワークは `ref_repositories/iLoRA/model/peft/tuners/gating.py` で定義された `Dense`, `MLP` などのクラスであり、`user_embeds` の次元に合わせた入力を持つ。
*   **本プロジェクトの実装**: `src/teacher/ilora_model.py` の `iLoRAModel` クラスの `forward` メソッドで、`self.rec_model` (`SASRec`) から得られたユーザー埋め込み (`user_embeds`) を `self.gating_network` に入力。`self.gating_network` は `MLPProjector` のインスタンスであり、`user_embeds` の次元に合わせた入力を持つ。ゲーティングネットワークは `F.softmax` を適用し、`num_lora_experts` 個のLoRAエキスパートに対するsoftmax重み (`gate_weights`) を出力。
*   **差分と再現性評価**:
    *   **再現性**: 参照実装と同様に、推薦モデルのユーザー埋め込みをゲーティングネットワークの入力とし、softmax重みを生成する機能が完全に再現されています。

#### 項目: LoRA パラメータの線形結合
*   **参照実装のロジック概要**: `ref_repositories/iLoRA/model/peft/tuners/moelora.py` の `MoeLoraModel` クラスの `Linear` モジュール内で、`gate_weights` を使って `lora_A` と `lora_B` の出力を重み付け結合。この `gate_weights` は `MInterface` の `forward` メソッドで `self.router(user_embeds)` から計算されたもの。LoRAアダプターの内部でゲーティングが行われ、各LoRAアダプターの出力がゲーティングされる。
*   **本プロジェクトの実装**: `src/teacher/moe_lora_model.py` の `Linear` クラスの `forward` メソッド内で、`gate_weights` を使って `lora_A` と `lora_B` の出力を重み付け結合。この `gate_weights` は `iLoRAModel` の `forward` メソッドで `self.gating_network(user_embeds)` から計算されたもの。LoRAアダプターの内部でゲーティングが行われ、各LoRAアダプターの出力がゲーティングされる。
*   **差分と再現性評価**:
    *   **再現性**: 参照実装のカスタムMoE LoRAレイヤーのロジックが、`src/teacher/moe_lora_model.py` にて完全に再現されています。

#### 項目: LLM への適用と次アイテム予測
*   **参照実装のロジック概要**: `ref_repositories/iLoRA/model/model_interface.py` の `wrap_emb` メソッドで、推薦モデルの埋め込みを特殊トークン (`[HistoryEmb]`, `[CansEmb]`, `[ItemEmb]`) に置き換えることでプロンプトエンジニアリングを行い、LLMは `inputs_embeds` を直接受け取る。
*   **本プロジェクトの実装**: `src/teacher/ilora_model.py` の `forward` メソッド内で、`input_ids` 内の特殊トークンの埋め込みを、`encode_items` および `encode_users` で生成した実際のアイテム/ユーザー埋め込みで置き換える。これにより、LLMへの入力が `inputs_embeds` を直接使用する形式になる。
*   **差分と再現性評価**:
    *   **再現性**: 参照実装と同様に、推薦モデルの埋め込みをLLMの入力埋め込みに直接注入する方式が再現されています。

#### 項目: 学習対象
*   **参照実装のロジック概要**: `ref_repositories/iLoRA/model/model_interface.py` の `configure_optimizers` で、`projector`、`router`、そして `llama_model` のパラメータ（`"gating" not in n` でフィルタリング）を学習対象としている。これにより、LoRAパラメータとゲーティングネットワークのパラメータのみが学習される。**また、ゲーティングネットワークの入力となる推薦モデル（SASRec）は事前に学習済みであり、そのパラメータは凍結されている。**損失関数は `MInterface` の `configure_loss` で `lm` (Language Model) ロス、つまり次アイテム予測のCross-Entropy Lossを使用。
*   **本プロジェクトの実装**: `iLoRAModel` の `__init__` でカスタムMoE LoRAレイヤーを使用しており、LoRAパラメータのみが学習可能に設定されている。`gating_network` と `output_layer` も `nn.Module` なので、これらも学習対象となる。**ゲーティングネットワークの入力となる推薦モデル（SASRec）は、事前に学習済みのチェックポイントからロードされ、そのパラメータは凍結される。**損失関数は `src/teacher/trainer_ilora.py` で `torch.nn.CrossEntropyLoss()` を使用。
*   **差分と再現性評価**:
    *   **再現性**: LoRAパラメータとゲーティングネットワークのパラメータが学習対象となり、LLMの基盤モデルがフリーズされるという点は再現されています。損失関数も次アイテム予測のCross-Entropy Lossを使用しており、この点も再現されています。**ゲーティングネットワークに利用される推薦モデルが事前に学習済みであり、凍結される点も参照実装と一致します。**

#### 項目: 教師出力
*   **参照実装のロジック概要**: `ref_repositories/iLoRA/model/model_interface.py` の `generate` メソッドや `validation_step`, `test_step` で生成された出力が、最終的にランキングスコアや埋め込みに変換されると推測される。
*   **本プロジェクトの実装**: `src/teacher/ilora_model.py` の `get_teacher_outputs` メソッドで `ranking_scores` (LLMの最終出力) と `embeddings` (LLMの最終隠れ状態) を明示的に返している。
*   **差分と再現性評価**:
    *   **再現性**: DLLM2Rec互換のランキングスコアと埋め込みを出力するという点は再現されています。
    *   **補足**: `DistillationTrainer`の評価時のモデルロード方法、`SASRecDataModule`の`_get_movie_id2name`メソッドの配置、評価メトリクス計算時のパディングアイテムIDの除外など、データ処理と評価に関する複数の問題が解決され、教師モデルの出力が蒸留パイプラインで正しく利用されるようになりました。

---

## 3. DLLM2Rec ロジックの再現性評価 (`src/distill/` vs `ref_repositories/DLLM2Rec/`)

### 3.1. DLLM2Rec の主要ロジック (00_overview.md より)

*   **教師ランキング＋スコアを用いたランキング蒸留 loss。**
*   **教師埋め込み vs 生徒埋め込みの距離を縮める埋め込み蒸留 loss。**
*   **CE loss（通常学習）＋ KD loss の合成。**

### 3.2. 評価項目と確認事項

以下の点について、本プロジェクトの `src/distill/kd_losses.py`, `src/distill/trainer_distill.py` および関連ファイルが、参照実装のロジックを再現しているかを確認してください。

*   **ランキング蒸留損失**:
    *   `RankingDistillationLoss` が、教師モデルと生徒モデルのランキングスコア（ロジット）を用いて、論文で提案されているランキング蒸留の数式を正しく実装しているか。
    *   温度パラメータ `T` の適用が適切か。
*   **埋め込み蒸留損失**:
    *   `EmbeddingDistillationLoss` が、教師モデルと生徒モデルの埋め込みを用いて、論文で提案されている埋め込み蒸留の数式（MSE または Cosine Similarity）を正しく実装しているか。
*   **損失の合成**:
    *   `DistillationTrainer` の `training_step` において、ランキング蒸留損失、埋め込み蒸留損失、および通常の Cross-Entropy Loss が、それぞれの重み (`ranking_loss_weight`, `embedding_loss_weight`, `ce_loss_weight`) に基づいて正しく合成されているか。
*   **蒸留サンプル選択ポリシー**:
    *   `selection_policy.py` で定義されているポリシー（例: `GroundTruthErrorPolicy`）が、蒸留に用いるサンプルを論文の意図に沿って選択しているか。

### 3.3. 評価結果

#### 項目: ランキング蒸留損失
*   **参照実装のロジック概要**: `ref_repositories/DLLM2Rec/main.py` では、LLMからの候補アイテム (`all_candidate`) と信頼度 (`llm_confidence`) を使用し、ポジション、共通アイテム、信頼度に基づく重み (`weight_rank`, `weight_com`, `weight_confidence`) を合成して、各候補アイテムに対するBCE損失を重み付け加算する形式。さらに、このBCE損失にはDRO損失が結合される。
*   **本プロジェクトの実装**: `src/distill/kd_losses.py` の `WeightedBCELoss` は、参照実装の重み付けBCE損失のロジックを再現し、各候補アイテムに対するBCE損失を計算する。`alpha > 0` の場合、このBCE損失にDRO損失が結合される。`src/distill/trainer_distill.py` では、`ranking_loss_weight` でこの結合された損失をスケーリングする。
*   **差分と再現性評価**:
    *   **損失の形式**: 本プロジェクトの `WeightedBCELoss` は、参照実装の重み付けBCE損失の形式を再現し、DRO損失も結合する。以前のKLダイバージェンスベースの `RankingDistillationLoss` は使用しない。
    *   **重み付け**: 参照実装と同様に、ポジション、共通アイテム、信頼度に基づく重み付けを `src/distill/trainer_distill.py` で計算し、`WeightedBCELoss` に渡すことで再現されている。
    *   **再現性**: 参照実装のランキング蒸留ロジック（重み付けBCE損失とDRO損失の結合）は、本プロジェクトで高いレベルで再現されている。さらに、`lam`パラメータによるランキング蒸留損失全体の重み付けも再現されました。

#### 項目: 埋め込み蒸留損失
*   **参照実装のロジック概要**: `ref_repositories/DLLM2Rec/main.py` の `GRU`, `Caser`, `SASRec` モデルの `forward` メソッド内で、`input_emb = input_emb + args.ed_weight * llm_emb` のように、LLMからの埋め込みを推薦モデルのアイテム埋め込みに直接加算することで、埋め込み空間を近づけようとしている。
*   **本プロジェクトの実装**: `src/student/models.py` の `SASRec` モデルの `_get_last_item_representation` メソッド内で、`item_embeddings` と `position_embeddings` を加算した `input_embeddings` に、教師モデルの埋め込み (`teacher_embeddings`) を直接加算するロジックを実装。`src/distill/kd_losses.py` の `EmbeddingDistillationLoss` は、この直接加算ロジックとは独立して、生徒モデルの出力埋め込みと教師モデルの出力埋め込みの距離を最小化する損失として引き続き使用される。
*   **差分と再現性評価**:
    *   **損失の適用方法**: 参照実装と同様に、LLM埋め込みを生徒モデルの入力埋め込みに直接注入するロジックを `src/student/models.py` に実装しました。これにより、DLLM2Recの埋め込み蒸留の主要なロジックが再現されました。`EmbeddingDistillationLoss` は、この直接加算の効果を補完する形で、出力埋め込みレベルでの蒸留を継続します。
    *   **再現性**: DLLM2Recの埋め込み蒸留の主要なロジックは高いレベルで再現されました。

#### 項目: DRO損失
*   **参照実装のロジック概要**: `ref_repositories/DLLM2Rec/main.py` では、アイテムの出現頻度に基づく傾向スコア (`ps`) を用いて、DRO損失 (`loss_dro`) を計算し、メインのBCE損失およびランキング蒸留損失に結合している。DRO損失の計算には `model_output * model_output` や `(model_output - 1) * (model_output - 1)` といったロジットの二乗項が含まれる。
*   **本プロジェクトの実装**: `src/distill/kd_losses.py` に `PropensityScoreCalculator` と `DROLoss` を実装。`PropensityScoreCalculator` は参照実装と同様に傾向スコアを計算。`DROLoss` は参照実装のDRO損失計算ロジックを再現し、数値安定性のために `exp` の引数と `torch.log` の引数をクランプしている。`src/distill/trainer_distill.py` では、メインのCross-Entropy損失と `WeightedBCELoss` (ランキング蒸留損失) の両方にこのDRO損失を結合している。
*   **差分と再現性評価**:
    *   **再現性**: 参照実装のDRO損失計算ロジックは、数値安定性のためのクランプ処理を除き、高いレベルで再現されている。傾向スコアの計算方法、DRO損失の数式、およびメインタスク損失とランキング蒸留損失への結合方法が一致している。

#### 項目: 損失の合成
*   **参照実装のロジック概要**: `ref_repositories/DLLM2Rec/main.py` の学習ループ内で、基本的なBCE損失に、dros loss (`args.alpha * torch.mean(loss_dro)`)、そしてランキング蒸留損失 (`args.lam * (loss_all_rd)`) が加算され、複数の損失が合成されている。
*   **本プロジェクトの実装**: `src/distill/trainer_distill.py` の `DistillationTrainer` クラスの `training_step` で、通常のCross-Entropy LossにDRO損失が結合され、さらに `ranking_kd_loss` (`ranking_loss_weight` で重み付け)、`embedding_kd_loss` (`embedding_loss_weight` で重み付け) が合成されている。
*   **差分と再現性評価**:
    *   **損失の種類**: 参照実装と同様に、メインタスク損失（Cross-Entropy）とランキング蒸留損失の両方にDRO損失が結合され、さらに埋め込み蒸留損失が合成される。
    *   **再現性**: 複数の損失を重み付けして合成するという点、およびDRO損失の適用箇所において、参照実装のロジックが再現されている。

#### 項目: 蒸留サンプル選択ポリシー
*   **参照実装のロジック概要**: `ref_repositories/DLLM2Rec/main.py` には、蒸留サンプルを選択する明示的なポリシーは確認できず、すべてのサンプルに対して損失を計算している。
*   **本プロジェクトの実装**: `src/distill/trainer_distill.py` の `training_step` で、`self.selection_policy.select(...)` を使用して蒸留サンプルを選択し、そのマスクをランキング蒸留損失と埋め込み蒸留損失の計算に適用している。`src/distill/selection_policy.py` には `AllSamplesPolicy`, `KLDivergencePolicy`, `GroundTruthErrorPolicy` が定義されている。
*   **差分と再現性評価**:
    *   **選択ポリシーの有無**: 本プロジェクトでは、`00_overview.md` の「将来拡張」の項目で言及されていた通り、Active Learning / Meta Learning のためのサンプル選択ポリシーを明示的に導入しています。これは参照実装にはない機能であり、本プロジェクト独自の拡張です。
    *   **再現性**: 参照実装には存在しない機能であるため、再現性という観点では「再現されていない」が、プロジェクトの目標（将来拡張）に沿った実装である。

---

## 5. 参照リポジトリの確認

`ref_repositories/DLLM2Rec/` の中身は詳細に確認済み。主要なコードは `main.py` に集約されており、損失関数や学習ループに関連する部分を把握した。
