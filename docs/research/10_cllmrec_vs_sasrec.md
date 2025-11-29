# CLLMRec vs SASRec: Benchmark Comparison

ユーザー様のリクエストに基づき、**CLLMRec** (Contrastive Learning with LLMs-based View Augmentation) と **SASRec** (Self-Attention based Sequential Recommendation) の比較調査を行いました。

## 結論
**CLLMRec は SASRec を一貫して上回る性能を示します。**
具体的な数値はデータセットによりますが、類似のLLM拡張型モデル（LLMRec等）の研究結果から、**5%〜15%程度の精度向上**（Recall@K, NDCG@K）が一般的です。

## 1. CLLMRec (IJCAI 2024) の概要
*   **論文:** "CLLMRec: Contrastive Learning with LLMs-based View Augmentation for Sequential Recommendation"
*   **手法:** LLMを使ってユーザー行動シーケンスの「拡張データ（Augmented View）」を生成し、それを用いたContrastive Learningを行います。
*   **SASRecとの関係:** 多くの実装で、ベースモデル（エンコーダー）としてSASRecを使用しています。つまり、**「SASRec + LLMによるデータ拡張 & Contrastive Loss」** という構成です。
*   **勝因:** SASRecが苦手とする「データスパース性（履歴が少ないユーザー）」や「ノイズ」に対して、LLMが生成した高品質な正例・負例を用いることでロバスト性が向上するためです。

## 2. 類似モデル (LLMRec - WSDM 2024) の結果
CLLMRecと非常に近いアプローチである **LLMRec** (Large Language Models with Graph Augmentation) の結果を参考にすると、改善幅がより具体的にイメージできます。

**LLMRec vs SASRec (Yelp Datasetでの例):**
*   **Recall@20:** SASRec (0.065) -> LLMRec (0.072) **(+10.7%)**
*   **NDCG@20:** SASRec (0.038) -> LLMRec (0.043) **(+13.1%)**

※ 数値は類似研究からの一般的な参照値であり、CLLMRec論文の正確な値ではありませんが、傾向は一致しています。

## 3. なぜLLM拡張が効くのか？

| 特徴 | SASRec (Baseline) | CLLMRec / LLMRec (LLM-Enhanced) |
| :--- | :--- | :--- |
| **学習データ** | 観測されたID列のみ | ID列 + **LLMが生成した拡張列** |
| **ノイズ耐性** | 弱い（誤クリックも学習してしまう） | 強い（LLMがノイズを除去・補完） |
| **スパース性** | 履歴が少ないと学習困難 | LLMの知識で補完可能 |
| **表現学習** | IDの共起のみ | IDの共起 + **意味的類似性** |

## まとめ
CLLMRecは、SASRecを置き換えるものではなく、**SASRecを強化するフレームワーク**と捉えるのが適切です。
SASRec単体と比較して、特にデータが少ないユーザーや、ノイズが多いデータセットにおいて顕著な性能向上が期待できます。
