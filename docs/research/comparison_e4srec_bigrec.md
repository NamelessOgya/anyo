# Comparison: E4SRec vs BigRec

## Overview
This document compares the implementation details of E4SRec and BigRec to identify the causes of the performance gap and propose improvements for E4SRec.

## Architectural Comparison

| Feature | E4SRec (Current) | BigRec | Impact |
| :--- | :--- | :--- | :--- |
| **Input Representation** | **Item Tokens** (IDs mapped to new vocab indices). Initially random. | **Item Names** (Text). "Toy Story", "Star Wars". | **BigRec Win**: Leverages LLM's pre-trained semantic knowledge immediately. E4SRec inputs are noise to a frozen LLM. |
| **Target Representation** | **Learned Embeddings**. Shared with input. Learned from scratch. | **Precomputed Embeddings**. Loaded from file. Likely semantic (avg token embs) or projected. | **BigRec Win**: Target is stable and semantically meaningful. E4SRec target is a moving target (learned). |
| **LLM Weights** | Frozen (except LoRA & Input/Output Embeddings). | Frozen (except LoRA). | Similar. |
| **Loss Function** | Sampled Softmax (Cross Entropy). | Full Cross Entropy (over fixed embeddings) or Text Gen. | BigRec's full softmax over fixed targets is more stable. |
| **Ensemble** | Dynamic Gating (SASRec + LLM). | Dynamic Gating (SASRec + LLM). | Similar (now that E4SRec has it). |
| **Inference** | Dot product with learned embeddings. | Dot product with fixed embeddings OR Text Generation. | BigRec supports both. |

## Key Findings

### 1. The Initialization Gap
The most critical difference is **Initialization**.
*   **BigRec**: Uses item names. The LLM `opt-125m` already knows what "Toy Story" is (semantically). The input is meaningful text. The target embeddings (if they are semantic) are also meaningful. The task is "Map User History (Text) -> Next Item (Text/Semantic)". The LLM is pre-trained for this.
*   **E4SRec**: Uses `<Item_1>`. This token is initialized randomly. The LLM sees "User History: <Random> <Random> ...". It has no idea what these tokens are. The LoRA adapter has to learn *everything* from scratch: the meaning of tokens, the user preference, and the mapping. Since the LLM is frozen, this is extremely difficult.

### 2. Dimension Mismatch Handling
*   **E4SRec**: Attempts to load SASRec embeddings but **skips** them because dimensions don't match (64 vs 768+). Falls back to random init.
*   **BigRec**: Loads `item_embeddings` that match the LLM dimension. These are likely pre-computed "Semantic Embeddings" (e.g., average of token embeddings for the item name).

## Improvement Proposals

### Proposal 1: Semantic Initialization for E4SRec (Recommended)
Instead of initializing E4SRec's item tokens randomly, initialize them using the **average token embeddings of the item names**.
*   **How**: In `factory.py`, iterate over all items. Tokenize their names. Get the embeddings from the frozen LLM. Average them. Assign this vector to the new item token in `input_embeddings`.
*   **Benefit**: The item tokens `<Item_1>` will immediately have the semantic meaning of "Toy Story". The frozen LLM can process them effectively.
*   **Effort**: Low. Modify `factory.py`.

### Proposal 2: Project SASRec Embeddings
If we want to leverage SASRec's geometric space:
*   **How**: Add a `Projector` (Linear layer) to map SASRec Embeddings (64) to LLM Embeddings (768+). Initialize E4SRec tokens with `Projector(SASRec_Emb)`. Train the Projector (or freeze it if pre-trained).
*   **Benefit**: Aligns with SASRec.
*   **Drawback**: SASRec space might not be semantically aligned with LLM space. Proposal 1 is safer for an LLM-based model.

### Proposal 3: Fix Target Embeddings
Instead of learning the target embeddings (which are shared with input), we could:
*   **Fix** the target embeddings to be the Semantic Embeddings (from Proposal 1).
*   Only learn the *Input* embeddings (or keep them fixed too and only learn LoRA).
*   This mimics BigRec's setup more closely.

## Conclusion
The performance gap is almost certainly due to **Random vs Semantic Initialization**. E4SRec is trying to learn a new language (Item IDs) with a frozen brain (LLM). BigRec speaks the brain's native language (Item Names).

**Next Step**: Implement **Proposal 1 (Semantic Initialization)** in `src/teacher/factory.py`.
