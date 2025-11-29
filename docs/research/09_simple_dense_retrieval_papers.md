# Simple Dense Retrieval with LLMs (anyo-like approaches)

`anyo` のように、LLMを純粋なエンコーダーとして使い、ベクトル近傍探索で推薦を行う「シンプルかつ強力」なアプローチを採用している代表的な論文を紹介します。

## 1. E4SRec (Efficient LLM for Sequential Recommendation)
**論文:** "Efficient Large Language Models for Sequential Recommendation" (2024)

*   **アプローチ:** `anyo` に非常に近いです。
*   **仕組み:**
    1.  LLM（Llamaなど）にユーザー履歴を入力します。
    2.  **最後の隠れ状態（Last Hidden State）**を取得します。
    3.  これを線形層（Projector）でアイテムEmbedding空間に射影します。
    4.  全アイテムとの内積（Dot Product）でスコアを計算し、Cross-Entropy Lossで学習します。
*   **特徴:**
    *   IDトークンを使わず、アイテムのテキスト情報のみを入力とします。
    *   推論時は事前に計算したアイテムEmbeddingとの近傍探索を行います。
    *   **LoRA** を用いて効率的に学習します。

## 2. CLLMRec (Contrastive Learning with LLMs for Recommendation)
**論文:** "Contrastive Learning with Large Language Models for Recommendation"

*   **アプローチ:** Dense Retrieval + Contrastive Learning
*   **仕組み:**
    *   LLMを使って、ユーザー履歴の「拡張（Augmentation）」を行います（例：履歴の一部を書き換える、要約するなど）。
    *   オリジナルと拡張版のEmbedding同士を近づける **Contrastive Loss** を追加します。
*   **anyoとの関連:**
    `anyo` の現在の実装（Sampled Softmax）は、実質的に「正例と負例（サンプリングされたアイテム）の距離を離す」という点で、Contrastive Learningの一種とみなせます。

## 3. SRA-CL (Semantic Retrieval Augmented Contrastive Learning)
**論文:** "Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation"

*   **アプローチ:** LLMによる意味的検索（Semantic Retrieval）を活用。
*   **仕組み:**
    *   LLMを使って、現在のユーザーと「意味的に似ているユーザー」や「意味的に似ているアイテム」を検索します。
    *   これらを正例（Positive Samples）としてContrastive Learningを行います。
*   **特徴:**
    単なるIDの一致だけでなく、LLMの知識（アイテムの意味）を活用して「似ているもの」を定義している点がユニークです。

## 4. LLaRA (Dense Retrieval Version)
**論文:** "Making Large Language Models A Better Foundation For Dense Retrieval"

*   **注意:** 前述の生成型LLaRAとは別の論文です。
*   **アプローチ:** LLMを汎用的なDense Retriever（検索器）として適応させる手法。
*   **仕組み:**
    *   クエリ（ユーザー）とドキュメント（アイテム）の両方をLLMでエンコードします。
    *   `[EOS]` トークンのEmbeddingを表現ベクトルとして使用します。
    *   Contrastive Lossで学習します。

## まとめ

`anyo` のアプローチ（LLM Encoder + Projection + MIPS）は、**E4SRec** や **LLaRA (Dense Retrieval版)** といった最新の研究トレンドと完全に一致しています。
これらは「生成の遅さ」という課題を解決しつつ、LLMの強力な文脈理解力を活かすための、現在最も合理的とされる解法です。
