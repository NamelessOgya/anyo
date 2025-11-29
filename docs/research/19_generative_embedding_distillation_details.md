# Generative Embedding Distillation: Technical Details

「Output Decision State (決定時の隠れ層)」を用いた蒸留手法の具体的なメカニズムと実装詳細です。

## 1. アーキテクチャ概要

この手法は、**「Teacher (RankLLM) の意思決定プロセス」** を **「Student (SASRec) のUser Embedding」** に焼き付けることを目的とします。

```mermaid
graph TD
    subgraph Student [Student: SASRec]
        History[User History] --> SASRecEncoder
        SASRecEncoder --> UserEmb[User Embedding u_sas]
        UserEmb --> ScoreCalc[Score Calculation]
        Items[Candidate Items] --> ScoreCalc
        ScoreCalc --> StudentScores[Student Scores]
    end

    subgraph Teacher [Teacher: RankLLM]
        Prompt[Prompt: History + Candidates] --> LLM
        LLM -- Deep Reasoning --> HiddenStates[Hidden States]
        HiddenStates --> Generation[Generation: '[ID_B] > [ID_A]...']
        
        subgraph Extraction [Decision State Extraction]
            Generation -- "1st Token '[ID_B]' Generated" --> Capture
            HiddenStates -- "Last Layer at '[ID_B]'" --> DecState[Decision State h_dec]
        end
    end

    DecState -- "Projector W_p" --> ProjState[Projected h_dec]
    
    %% Distillation Losses
    ProjState <-->|Embedding Loss (MSE)| UserEmb
    Generation <-->|Ranking Loss (ListNet)| StudentScores
```

## 2. 詳細プロセス (Step-by-Step)

### Step 1: 候補取得 (Candidate Retrieval)
Student (SASRec) を使い、現在のユーザー履歴に基づいて候補アイテム（例: 20件）を取得します。
*   Candidates: $\{i_1, i_2, ..., i_{20}\}$

### Step 2: Teacher推論 & 意思決定 (Teacher Forward)
Teacher (RankLLM) にプロンプトを入力し、推論を実行します。

*   **Prompt:** `User History: [...]. Candidates: [1] i_1, [2] i_2, ... Rank them.`
*   **Deep Reasoning:** LLMのSelf-Attention層が、履歴と候補アイテム間の複雑な関係性を分析します。
*   **Decision (意思決定):** LLMが「アイテム $i_5$ (ID: [5]) が最も適切である」と判断します。
*   **Generation:** 次のトークンとして `[5]` を生成します。

### Step 3: Hidden State 抽出 (Extraction)
ここが核心です。LLMが `[5]` を生成した **その瞬間（ステップ）** の最終隠れ層ベクトルを取得します。

*   **Target:** `outputs.last_hidden_state` (Batch, SeqLen, HiddenDim)
*   **Position:** 生成されたトークン `[5]` に対応する位置。
*   **Vector ($h_{dec}$):** このベクトルには、**「なぜ他の候補ではなく $i_5$ を選んだのか」** という文脈情報（推論の結果）が凝縮されています。

### Step 4: Student 推論 (Student Forward)
Student (SASRec) も同じユーザー履歴から User Embedding ($u_{sas}$) を生成します。

### Step 5: 蒸留 (Distillation)
2つのLossを計算し、Studentを更新します。

1.  **Embedding Loss ($L_{emb}$):**
    *   Teacherの「意思決定ベクトル ($h_{dec}$）」と、Studentの「User Embedding ($u_{sas}$）」を近づけます。
    *   次元数が異なるため、Projector ($W_p$) を通します。
    *   $$ L_{emb} = || W_p(h_{dec}) - u_{sas} ||^2 $$
    *   **意味:** Studentは「Teacherと同じ意思決定」ができるような「ユーザー表現」を獲得します。

2.  **Ranking Loss ($L_{rank}$):**
    *   Teacherが生成した順序（例: $i_5 > i_2 > ...$）を正解とし、Studentのスコア分布を近づけます。
    *   **意味:** 出力結果の整合性を保証します。

## 3. Pseudo-Code

```python
# Training Loop
for batch in dataloader:
    # 1. Studentで候補取得 (省略)
    candidates = get_candidates(student_model, batch)
    
    # 2. Teacherで推論 (RankLLM)
    prompt = create_prompt(batch['history'], candidates)
    teacher_inputs = tokenizer(prompt, return_tensors="pt")
    
    # generate() を実行しつつ、hidden_statesを取得
    with torch.no_grad():
        outputs = teacher_model.generate(
            **teacher_inputs,
            max_new_tokens=10, # 最初の数トークンだけでOK
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # 3. Decision State ($h_{dec}$) の抽出
    # 生成された最初のトークン（1位のアイテムID）に対応するHidden State
    # sequences: [Prompt, Generated_Token_1, ...]
    # hidden_states: tuple of (layers) at each step
    
    # 最初の生成ステップ(step=0)の、最終層(-1)のHidden State
    # shape: (Batch, HiddenDim)
    h_dec = outputs.hidden_states[0][-1][:, -1, :] 
    
    # 4. Student Forward
    u_sas = student_model.get_user_embedding(batch['history']) # (Batch, EmbDim)
    student_scores = student_model.predict(u_sas, candidates)
    
    # 5. Loss Calculation
    # Projectorで次元合わせ
    h_dec_proj = projector(h_dec)
    
    # Embedding Loss
    loss_emb = F.mse_loss(u_sas, h_dec_proj)
    
    # Ranking Loss (Teacherの順位に基づく)
    teacher_rank = parse_ranking(outputs.sequences)
    loss_rank = listnet_loss(student_scores, teacher_rank)
    
    # Total Loss
    loss = loss_rank + lambda_emb * loss_emb
    
    loss.backward()
    optimizer.step()
```

## 4. なぜこれが「Deep Reasoning」の蒸留になるのか？

通常のEmbedding蒸留（Contrastive Learningなど）は、「似たアイテムは近くに」という **静的な類似性** を学習します。

対してこの手法は、**「特定の文脈（履歴+候補）において、なぜそのアイテムが選ばれたか」** という **動的な意思決定プロセス** を学習します。
Teacherの $h_{dec}$ は、Attention層を経て「履歴」と「候補」を深く比較検討した結果のベクトルです。これをStudentが模倣することで、StudentのUser Embeddingは単なる「履歴の平均」を超えて、**「推論結果を先取りしたベクトル」** へと進化します。
