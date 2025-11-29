# Generative Reranker with Hidden State Distillation (Final)

ユーザー様のご指摘は完全に正しいです。
「プロンプト読込直後（入力）」の隠れ層は、まだ推論（生成）を行っていないため、生成結果（Ranking）とは独立しています。これでは「Deep Reasoningの蒸留」になりません。

**修正案: 「Output Decision State (決定時の隠れ層)」を蒸留します。**

## アルゴリズム詳細

### 1. Teacher (RankLLM) の推論プロセス
1.  プロンプト入力: `Rank these items: [A], [B], [C]...`
2.  **Deep Reasoning (内部計算):** Attention層で全アイテムを比較検討。
3.  **生成 (Output):** `[B]` (1位) > `[A]` (2位) ...
4.  **Target Embeddingの抽出:**
    *   LLMが **1位のアイテムトークン `[B]` を生成したステップの最終隠れ層ベクトル ($h_{dec}$) ** を取得します。
    *   この $h_{dec}$ こそが、「比較検討の結果、Bが1位である」と判断した **推論の結晶** です。

### 2. Student (SASRec) の学習プロセス
Studentは「User Embedding ($u_{sas}$）」を使って、この「Teacherの決定 ($h_{dec}$)」を予測します。

*   **Embedding Loss ($L_{emb}$):**
    *   $L_{emb} = || W_p(h_{dec}) - u_{sas} ||^2$
    *   StudentのUser Embeddingを、Teacherの「意思決定ベクトル」に近づけます。
    *   これにより、Studentは「Teacherがなぜそれを選んだか」という文脈（Reasoning）をベクトル空間で模倣します。

*   **Ranking Loss ($L_{rank}$):**
    *   Teacherの出力順位に基づくスコア分布を、Studentのスコア分布 ($u_{sas} \cdot i_{all}$) で模倣します。

## この修正による変化
*   **Before (Input State):** 「入力文脈」を模倣。推論結果とは無関係。
*   **After (Output State):** **「推論結果（意思決定）」を模倣。** 推薦結果とEmbeddingが完全に連動します。

## 実装上のポイント
*   `RankListwiseOSLLM` の `generate` ループ内で、最初のアイテムIDトークンが生成された瞬間の `hidden_states` をキャプチャする必要があります。
*   これは `model.generate()` の `return_dict_in_generate=True, output_hidden_states=True` オプションで取得可能です。
