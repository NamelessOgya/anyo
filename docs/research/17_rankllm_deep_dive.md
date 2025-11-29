# RankLLM Deep Dive: Methodology & Implementation

ユーザー様のリクエストに基づき、`RankLLM` (Castorini/RankLLM) のコードを解析し、その手法（Listwise Reranking）の詳細をまとめました。

## 1. Listwise Reranking の仕組み

RankLLMは、複数のアイテム（Passage）を一度にLLMに入力し、**「それらの相対的な順序（Permutation）」を直接生成させる** 手法です。

### プロンプト構造 (`rank_zephyr_template.yaml`)

LLMには以下のような構造化されたプロンプトが渡されます。

```text
[System Message]
You are RankLLM, an intelligent assistant that can rank passages...

[Prefix]
I will provide you with 20 passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: "iPhone 15 Pro Max".

[Body] (候補リスト)
[1] The iPhone 15 Pro Max features a titanium design...
[2] Samsung Galaxy S24 Ultra is a strong competitor...
...
[20] Apple released a new case for iPhone...

[Suffix] (指示)
Search Query: "iPhone 15 Pro Max".
Rank the 20 passages above based on their relevance...
The output format should be [] > [], e.g., [2] > [1], Answer concisely...
```

### LLMの出力
LLMは以下のような文字列を生成します。
```text
[1] > [5] > [2] > ... > [20]
```
これを解析（Parse）して、実際のリストを並び替えます。

## 2. Sliding Window による全件ソート

LLMのコンテキスト長（入力可能なトークン数）には限りがあるため、100件や1000件の候補を一度に並び替えることはできません。
RankLLMは **Sliding Window** アルゴリズムを使って、これを解決しています。

### アルゴリズムの挙動 (`listwise_rankllm.py`)

特徴的なのは、**「後ろから前へ（Back-to-Front）」** 処理を行う点です。

**設定例:**
*   候補数: 100件
*   Window Size: 20件
*   Stride (ずらす幅): 10件

**ステップ:**
1.  **Window 1 (Rank 80-100):**
    *   下位20件をLLMに入力し、並び替える。
    *   良いアイテムがWindow内の上位（80位付近）に来る。
2.  **Window 2 (Rank 70-90):**
    *   Windowを10件ずらす。
    *   先ほどの「勝ち上がり組」と、その上の10件を戦わせる。
    *   ここで勝ったアイテムはさらに上（70位付近）に行く。
3.  **... (繰り返し) ...**
4.  **Final Window (Rank 0-20):**
    *   最後にトップ20件を並び替えて、最終的なランキングが確定する。

この「バブルソート」のような仕組みにより、下位に埋もれていた高関連アイテムも、段階的に上位に浮上することができます。

## 3. anyo への応用（蒸留用Teacherとして）

この手法は、`anyo` のTeacherモデルとして非常に適しています。

1.  **高品質な順序情報:** Pointwise（1件ずつ採点）よりも、Listwise（比較検討）の方が、LLMの推論能力を活かした高精度なランキングが作れます。
2.  **実装の容易さ:** `RankLLM` のプロンプトテンプレートとSliding Windowロジックは、そのまま流用可能です。
3.  **Single GPU対応:** Window Sizeを調整（例: 20件）すれば、7Bモデル + QLoRA (Single GPU) でも十分に動作します。

**提案するフロー:**
1.  Student (SASRec) で候補を100件取得。
2.  Teacher (7B QLoRA) で RankLLM方式のSliding Window Rerankingを実行。
3.  得られた「並び替え済みリスト（順位）」を正解として、Studentに蒸留する。
