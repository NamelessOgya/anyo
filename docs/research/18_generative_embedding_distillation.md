# Generative Reranker with Hidden State Distillation (Refined)

ユーザー様のご指摘通り、Listwise Reranker (Cross-Encoder) から「静的なアイテムEmbedding」を蒸留しても、それは文脈（Ranking）を含まない単なる意味表現になりがちです。

そこで、**「User Embedding (Query Context) の蒸留」** を主軸に置いた手法に修正します。

## 修正版アルゴリズム

### 1. LLMを使ってTextを生成する (Reasoning)
*   **手法:** RankLLM (Listwise Reranking)
*   **役割:** 候補アイテム群を比較検討し、最適な順序を決定する「推論」を担当。

### 2. アイテムのスコアが計算可能である
*   **手法:** Rank-based Score または Logits
*   **役割:** Studentの出力スコア分布をTeacher（順位）に近づける **Ranking Distillation** の教師信号。

### 3. 蒸留可能なEmbeddingの獲得 (User Representation Distillation)
*   **課題:** RankLLMは「User x Item」の相互作用を見て順位を決めるため、静的な「Item Embedding」には順位情報は含まれません。
*   **解決策:** **「User Embedding（クエリ表現）」を蒸留します。**
    *   **Teacher (LLM):** プロンプト（履歴 + 候補）を読み込んだ直後、または生成開始直前の **最終隠れ層ベクトル ($h_{ctx}$)** を取得します。これには「このユーザーが何を求めているか」という **推論済みの文脈情報** が凝縮されています。
    *   **Student (SASRec):** 自身のUser Embedding ($u_{sas}$) を、この $h_{ctx}$ に近づけるように学習します。
    *   **Loss:** $L_{emb} = || W_p(h_{ctx}) - u_{sas} ||^2$

### なぜこれで解決するのか？
*   Student (SASRec) のスコアは $Score = u_{sas} \cdot i_{item}$ で決まります。
*   Teacherの「推論結果（Ranking）」は、Teacherの「文脈理解 ($h_{ctx}$）」に基づいています。
*   Studentの $u_{sas}$ が Teacherの $h_{ctx}$ を模倣することで、**「Teacherと同じ視点（文脈理解）」** を獲得できます。
*   結果として、Studentは「Teacherが良いと判断しそうなアイテム」に対して高いスコアを出せるようになります。

## 最終的な蒸留構成

$$ L_{total} = \lambda_1 L_{rank} (Score_{student}, Rank_{teacher}) + \lambda_2 L_{emb} (u_{student}, h_{teacher\_ctx}) $$

1.  **$L_{rank}$:** 出力結果（順位）の整合性を担保。
2.  **$L_{emb}$:** 内部表現（ユーザー理解）の整合性を担保。

この構成であれば、Embedding Distillationが独立することなく、推薦結果（Ranking）を支える「文脈理解」の向上に直接寄与します。
