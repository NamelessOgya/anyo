# Proposal: Rationale-Distilled Representation (RDR)

## 1. 概要 (Overview)

**Rationale-Distilled Representation (RDR)** は、Teacherモデル（LLM）が持つ「推薦理由（Rationale）を言語化する能力」を、Studentモデル（SASRecなどのIDベースモデル）の「潜在表現空間（Latent Space）」に蒸留する手法です。

従来の蒸留（Knowledge Distillation）は、Teacherが出力する「確率分布（Logits）」をStudentに模倣させるものでした。これは「**何を（What）** 推薦すべきか」という情報の転移に留まります。
対してRDRは、Teacherに「**なぜ（Why）** そのアイテムを推薦するのか」という理由を生成させ、その理由に対応するベクトル表現にStudentの表現を近づけることで、より深い意味的理解（Semantic Understanding）をStudentに与えることを目的とします。

---

## 2. 背景と動機 (Motivation)

### 2.1. IDベースモデルの限界
SASRecなどのIDベースモデルは、アイテムIDの共起関係から学習するため、協調フィルタリング（CF）効果は高いですが、「アイテムの内容（コンテンツ）」や「推薦の論理的根拠」を理解しているわけではありません。そのため、学習データにない組み合わせ（Cold Start）や、複雑な文脈（「悲しい気分の時に見る映画」など）に対応するのが苦手です。

### 2.2. LLMの強み
LLMは豊富な一般的知識を持っており、アイテムのメタデータ（ジャンル、監督、あらすじ）や、ユーザーの嗜好パターン（「このユーザーはSF好き」など）を言語的に理解しています。この「推論プロセス」こそが、LLMがIDモデルより優れている本質的な理由の一つです。

### 2.3. 提案の核心
「LLMの推論プロセス（Rationale）をベクトル化し、それを正解ベクトル（Target）としてStudentに学習させれば、Studentのベクトル空間も意味的な構造を持つようになるのではないか？」というのが本提案の核心です。

---

## 3. アーキテクチャと手順 (Methodology)

本手法は、大きく3つのフェーズで構成されます。

### Phase 1: Rationale Generation (理由生成)
学習データセットの各サンプル $(u, i, h)$ （ユーザー $u$, 正解アイテム $i$, 履歴 $h$）に対して、Teacherモデル（LLM）を用いて推薦理由 $R$ を生成します。

*   **Prompt例:**
    ```text
    User History: [Toy Story, Finding Nemo, Monsters, Inc.]
    Target Item: The Incredibles
    Question: Why is 'The Incredibles' a good recommendation for this user?
    Answer: Because the user clearly enjoys Pixar animations with themes of family and adventure. 'The Incredibles' shares these characteristics and is also directed by Brad Bird...
    ```
*   **処理:** この生成は計算コストが高いため、**オフラインで一度だけ実行**し、データセットとして保存します。

### Phase 2: Rationale Encoding (理由のベクトル化)
生成されたテキスト $R$ を、固定長のベクトル $v_{rationale}$ に変換します。

*   **Encoderの選択肢:**
    1.  **TeacherのEmbedding:** Teacherモデル（iLoRA）のEmbedding層や中間層を使用する。Teacher自身の知識空間に合わせるため、最も整合性が高い。
    2.  **外部Encoder:** SBERT (Sentence-BERT) や E5 などの汎用テキストEncoderを使用する。計算が軽量で、一般的な意味空間を利用できる。
*   **推奨:** まずは実装が容易で品質が保証されている **SBERT (all-MiniLM-L6-v2等)** の利用を推奨します。

### Phase 3: Distillation Training (蒸留学習)
Studentモデルの学習時に、通常のNext Item Prediction Loss ($L_{CE}$) に加えて、Rationale Distillation Loss ($L_{RDR}$) を追加します。

*   **Studentの出力:** 履歴 $h$ を入力した際の、Studentの最終隠れ層ベクトル $v_{student} = \text{SASRec}(h)$。
*   **Loss関数:**
    $$L_{RDR} = 1 - \cos(v_{student}, v_{rationale})$$
    （コサイン類似度を最大化、つまり距離を最小化する）
*   **全体のLoss:**
    $$L_{total} = L_{CE} + \lambda \cdot L_{RDR}$$

---

## 4. 学術的新規性と貢献 (Academic Contribution)

1.  **Beyond Logits Distillation:**
    単なるスコアの蒸留ではなく、「推論過程（Reasoning）」の蒸留という新しいパラダイムを推薦システムに導入しています。これはNLP分野の "Chain-of-Thought Distillation" をRecSysに応用した先駆的な試みと言えます。

2.  **Semantic Regularization:**
    IDベースの疎な空間に、言語的な意味空間（Semantic Space）の制約を与えることで、強力な正則化（Regularization）として機能します。これにより、過学習の抑制や汎化性能の向上が期待できます。

3.  **Explainability Potential:**
    Studentのベクトル空間が「理由」の空間と整列（Align）するため、Studentのベクトルを用いて類似する「理由テキスト」を検索・生成することが可能になり、IDモデルでありながら説明可能性（Explainability）を持つ可能性があります。

---

## 5. 実装へのステップ (Implementation Steps)

### Step 1: データセット作成 (Offline)
*   既存の学習データからサブセット（例: 10%）を抽出。
*   LLM (Gemma-2b-it等) を用いて、各サンプルのRationaleを生成し、JSONファイルに保存。
    *   `{"user_id": 1, "item_id": 101, "rationale": "..."}`

### Step 2: Vector Store作成 (Offline)
*   生成されたRationaleをSBERT等でベクトル化し、`torch.Tensor` として保存（またはHDF5）。

### Step 3: Trainerの拡張
*   `DistillationTrainer` を改修し、バッチに対応する $v_{rationale}$ をロードする仕組みを追加。
*   $L_{RDR}$ の計算ロジックを追加。

### Step 4: 比較実験
*   Baseline (SASRec) vs Baseline + Logits KD vs Baseline + RDR で精度比較。

---

## 6. 懸念点と対策

*   **コスト:** 全データに対するRationale生成はコストが高い。
    *   **対策:** 学習データの数%のみに適用する、または「自信がないサンプル」のみに適用する（UAMDとの組み合わせ）。
*   **幻覚 (Hallucination):** LLMが適当な理由をでっち上げる可能性がある。
    *   **対策:** 理由の正しさよりも「意味的な関連付け」が行われること自体に価値があると割り切る。または、Promptで「事実に基づいて」と制約する。
