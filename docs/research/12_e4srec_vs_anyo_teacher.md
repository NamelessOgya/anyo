# E4SRec vs anyo Teacher Model: Detailed Code Comparison

ユーザー様のリクエストに基づき、`E4SRec` (HestiaSky/E4SRec) と `anyo` の教師モデル (`iLoRAModel`) のコードレベルでの比較を行いました。

## 結論
両者は **「SASRecのID EmbeddingをLLMに入力し、Dense Retrievalを行う」** という基本コンセプトは共通していますが、**`anyo` の方がアーキテクチャとして高度（複雑）**です。
具体的には、`anyo` は **MoE-LoRA (Mixture-of-Experts)** を採用しており、ユーザーごとにLoRAアダプターを動的に切り替える点が最大の違いです。

## 詳細比較表

| 特徴 | E4SRec (HestiaSky/E4SRec) | anyo (Current Project) |
| :--- | :--- | :--- |
| **ベースモデル** | LlamaModel (Decoder only) | AutoModel (Decoder only) |
| **ID Injection** | `nn.Embedding` (Frozen) -> `nn.Linear` | `SASRec` (Trainable/Frozen) -> `MLPProjector` |
| **Adapter技術** | **Standard LoRA** (Peft) | **MoE-LoRA** (Custom Implementation) |
| **Prompting** | `[Instruct] + [InputEmbeds] + [Response]` (連結) | `[HistoryEmb]` プレースホルダー置換 |
| **出力層** | `nn.Linear` (Hidden -> NumItems) | `Dot Product` (Hidden @ ItemEmbeds.T) |
| **学習対象** | LoRA, Input Projector, Output Head | MoE-LoRA, Projector, Gating Network, (SASRec) |

## コードレベルの分析

### 1. ID Injection (入力部分)

**E4SRec (`model.py`):**
シンプルに、事前学習済みのEmbeddingをLinear層で射影し、トークンEmbeddingと結合（Concatenate）しています。
```python
# E4SRec
self.input_embeds = nn.Embedding.from_pretrained(..., freeze=True)
self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)

# Forward
items = self.input_proj(self.input_embeds(inputs[:, 1:]))
inputs = torch.cat([instruct_embeds, items, response_embeds], dim=1)
```

**anyo (`ilora_model.py`):**
SASRecモデル全体を保持し、その出力をMLP（多層パーセプトロン）で射影しています。さらに、プロンプト内の特定のトークン `[HistoryEmb]` を置換する方式をとっています。
```python
# anyo
full_sequence_rec_embs = self.rec_model.get_full_sequence_representations(...)
his_item_embeds = self.projector(full_sequence_rec_embs)

# Forward (Placeholder Replacement)
history_emb_mask = (input_ids == his_token_id)
modified_input_embeds[history_emb_mask] = his_item_embeds[...]
```

### 2. Adapter Architecture (LLM内部)

**E4SRec:**
Hugging Face PEFTライブラリの標準的なLoRAを使用しています。
```python
# E4SRec
peft_config = LoraConfig(task_type='FEATURE_EXTRACTION', r=self.args['lora_r'], ...)
self.llama_model = get_peft_model(self.llama_model, peft_config)
```

**anyo:**
独自の `MoeLoraModel` を実装し、ユーザーの特徴（Embedding）に応じて複数のLoRAエキスパートを切り替える **Gating Network** を持っています。
```python
# anyo
self.llm = MoeLoraModel(..., num_lora_experts=num_lora_experts)
self.gating_network = MLPProjector(...)

# Forward
gate_weights = F.softmax(self.gating_network(user_embeds), dim=-1)
self.llm.gate_weights.append(gate_weights)
```

### 3. Retrieval (出力部分)

**E4SRec:**
単純な全結合層（Linear）でクラス分類を行っています。これは「全アイテムに対するスコア計算」です。
```python
# E4SRec
self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False)
pooled_logits = self.score(pooled_output)
```

**anyo:**
LLMの出力と、射影されたアイテムEmbeddingとの内積（Dot Product）を計算しています。数式的にはE4SRecのLinear層と同じ（重み行列＝アイテムEmbedding行列と見なせば）ですが、実装としては「検索（Retrieval）」を意識した形になっています。

## まとめ
*   **E4SRec** は「シンプル・高速・効率的」を追求した実装です。標準的なLoRAとLinear層のみを使用し、実装が非常にクリーンです。
*   **anyo** は「表現力・パーソナライズ」を追求した実装です。MoE-LoRAにより、ユーザーの文脈に応じてLLMの振る舞いを動的に変化させることができます。また、SASRecモデルをそのまま組み込んでいるため、SASRec側のFine-tuningも可能です。

E4SRecの方が「ベースライン」として適切であり、anyoはその「発展版（MoE拡張版）」という位置づけになります。
