# Alternative Embedding-Based LLM Recommendation Models

ユーザーの要件（LLMを活用し、最終的にEmbeddingを出力して内積で推薦を行うモデル）に合致する、iLoRA以外の有力なアプローチをまとめました。これらは「DLL2REC（Distilled LLM to Rec）」の代替案として検討可能です。

## 1. Contriever-Rec / Dense Retrieval with LLM
LLMを「テキスト生成器」としてではなく、「強力なエンコーダー」として使用し、ユーザー履歴とアイテムを同じベクトル空間にマッピングするアプローチです。

### iLoRAとの比較
| 特徴 | iLoRA (Current) | Contriever-Rec |
| :--- | :--- | :--- |
| **アーキテクチャ** | **Asymmetric (非対称)**<br>User: LLM + LoRA<br>Item: SASRec Embedding (ID) | **Symmetric (対称)**<br>User: LLM Encoder<br>Item: LLM Encoder (Text) |
| **空間の基準** | **SASRecの空間**に合わせる (Distillation) | **LLMの言語空間** (または共通の新しい空間) |
| **アイテム表現** | **IDベース** (SASRecの学習済みベクトル) | **テキストベース** (タイトル・説明文のEmbedding) |
| **学習難易度** | **高** (異なる空間の強制アライメントが必要) | **中** (Contrastive Lossで自然に近づける) |
| **Cold Start** | 苦手 (IDがないと扱えない) | **得意** (テキストさえあればEmbeddingを作れる) |

*   **議論**: iLoRAが苦戦している「Projectorの学習（空間の翻訳）」が不要になります。最初から同じLLMエンコーダーを使うため、空間のズレが少なく、学習がスムーズです。ただし、ID固有の協調フィルタリング情報は失われる可能性があります。

## 2. LLM2Vec / Prompt-based Embedding
Decoder-only LLM（Llama, Qwenなど）は本来「次の単語予測」が得意ですが、これをEmbeddingモデルとして転用する手法です。

### iLoRAとの比較
| 特徴 | iLoRA (Current) | LLM2Vec |
| :--- | :--- | :--- |
| **出力形式** | **Last Token Hidden State** (Next Token Prediction用) | **Weighted Mean Pooling** (全トークンの平均) |
| **目的関数** | Ranking Loss (Softmax) + Distillation | Contrastive Loss (InfoNCE) |
| **コンテキスト** | Causal Mask (過去のみ参照) | **Bidirectional Attention** (双方向参照が可能になる工夫あり) |

*   **議論**: iLoRAは「次のアイテム」を予測するために「最後のトークン」だけを使いますが、LLM2Vecは「履歴全体」の意味をベクトル化します。推薦タスクにおいては、ユーザーの興味を一点（次のアイテム）に絞るiLoRAの方が適している場合もありますが、長期的な興味のモデリングにはLLM2Vecのアプローチが有効かもしれません。

## 3. E4SRec (Efficient Efficient LLM for Sequential Recommendation)
IDベースの推薦モデル（SASRecなど）の効率性と、LLMの意味理解能力を融合させたモデルです。

### アーキテクチャ詳細 (Deep Dive)
E4SRecの最大の特徴は、**「ID Embedding」と「Text Embedding」をLLMの入力層で混ぜる** ことです。

1.  **入力層 (Input Layer)**:
    *   **ID Embedding**: アイテムIDごとに学習可能なベクトルを用意（SASRecと同じ）。
    *   **Text Embedding**: アイテムのタイトルなどをLLMのトークナイザーでID化し、LLMのEmbedding層を通したもの。
    *   **Fusion**: これらをどう混ぜるかが重要です。E4SRecでは、プロンプトの中に `<item_id_token>` のようなプレースホルダーを作り、そこに対応するID Embeddingを注入します。
        *   例: `User History: <item_1> (Movie A), <item_2> (Movie B) -> Predict Next: `

2.  **バックボーン (Backbone)**:
    *   Llama-2 などのDecoder-only LLMを使用。
    *   **LoRA (Low-Rank Adaptation)**: 全パラメータは重すぎるので、LoRAアダプターのみを学習させます。これにより、LLMが「IDの意味」と「テキストの意味」の関係性を学習します。

3.  **出力層 (Output Layer)**:
    *   通常のLLMは「次の単語（トークン）」を予測しますが、E4SRecは **「次のアイテムID」** を予測したいです。
    *   そこで、LLMの最後の隠れ層（Last Hidden State）を取り出し、**全アイテムのID Embeddingとの内積（スコア）** を計算します。
    *   **Loss**: Cross Entropy Loss (または Sampled Softmax)。

### iLoRAとの決定的な違い
*   **iLoRA**: 「SASRec」と「LLM」が別々のモデルとして存在し、Projectorで無理やり繋いでいます（疎結合）。
*   **E4SRec**: 「LLMの中にID Embeddingを組み込む」ことで、LLM自体をRecSysとして拡張しています（密結合）。

### なぜE4SRecの方が良い可能性があるか？
1.  **情報のロスが少ない**: iLoRAのProjectorは「SASRecの出力」という圧縮された情報しか見れませんが、E4SRecは「生のID」と「生のテキスト」の両方をLLMの強力なAttention機構で処理できます。
2.  **学習が素直**: 「Projectorの翻訳」という難しいタスクがなく、最初から「IDとテキストを見て次を当てる」という自然なタスク設定になっています。
3.  **ID情報の活用**: テキストだけでは区別しにくいアイテム（「マトリックス」と「マトリックス リローデッド」など）も、ID Embeddingがあることで明確に区別できます。

### 実装のポイント
*   LLMの語彙（Vocabulary）を拡張して、アイテム数分のトークンを追加するイメージです。
*   `resize_token_embeddings` で語彙サイズを増やし、その増えた部分のEmbeddingを「アイテムEmbedding」として学習させます。

## 4. LlamaRec (Two-Stage Approach)
「全アイテムからの検索（Retrieval）」と「詳細な並び替え（Reranking）」を分けるアプローチです。

*   **仕組み**:
    1.  **Retrieval**: 軽量なモデル（SASRecなど）で候補を数百件に絞る。
    2.  **Reranking**: LLMに「履歴」と「候補アイテム」を入力し、各候補のスコア（ランク）を出力させる。
*   **Embedding化**:
    *   このReranking部分を、LLMのLast Hidden Stateを使った「Pointwise Scoring（内積）」に置き換えることで、高速化を図る変種もあります。

## 推奨される代替案
もしiLoRA（SASRec空間への蒸留）がうまくいかない場合、最も有望なのは **「1. Contriever-Rec / Dense Retrieval」** アプローチです。

*   **理由**:
    *   構造がシンプル（Dual Encoder）。
    *   「SASRecの空間」という制約に縛られず、LLMが持つリッチな言語空間をフル活用できる。
    *   学習が安定しやすい（Contrastive Lossは実績豊富）。

iLoRAのコードベースを活かすなら、**「Projectorの向きを変える（LLM出力 → 共通空間 ← Item Embedding）」** だけで実装可能です。
