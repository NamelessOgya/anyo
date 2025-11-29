# BIGRec Deep Dive & Implementation Plan

## Overview
**BIGRec** ([A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems](https://arxiv.org/abs/2308.08434)) proposes a paradigm where an LLM is fine-tuned to directly generate the target item's title given a user's interaction history.

## Methodology
1.  **Instruction Tuning:** The model is trained on `(Instruction, Input, Output)` triples.
    -   **Instruction:** "Recommend the next item for the user."
    -   **Input:** Sequence of historical item titles (e.g., "Item A, Item B, Item C").
    -   **Output:** Target item title (e.g., "Item D").
2.  **LoRA Fine-tuning:** Uses Low-Rank Adaptation (LoRA) to efficiently fine-tune large models (e.g., Llama, Qwen) on consumer hardware.
3.  **Inference:** The model generates the title of the next item. This title is then "grounded" to an item ID in the catalog (usually by exact match or similarity search).

## Comparison with Existing `anyo` Approaches
| Feature | iLoRA (Current) | RankLLM (Teacher) | BIGRec (New) |
| :--- | :--- | :--- | :--- |
| **Role** | Teacher (Soft) | Teacher (Reranker) | **Generative Recommender** |
| **Input** | Embeddings (SASRec) | Text (Candidates) | **Text (History)** |
| **Output** | Embeddings / Logits | Ranking / Scores | **Text (Item Title)** |
| **Training** | MoE-LoRA (Hybrid) | Frozen / Fine-tuned | **Instruction Tuning (LoRA)** |
| **Inference** | Score all items | Rerank top-K | **Generate Next Item** |

## Implementation Plan for `anyo`

To integrate BIGRec into `anyo` while maintaining consistency with the existing PyTorch Lightning infrastructure:

### 1. Model (`src/teacher/bigrec_model.py`)
Create a `BigRecModel` class inheriting from `pl.LightningModule`.
-   **Base:** `AutoModelForCausalLM` (e.g., Qwen/Llama).
-   **PEFT:** Apply `LoraConfig` via `get_peft_model`.
-   **Training Step:**
    -   Construct prompts from batch data.
    -   Tokenize prompts and targets.
    -   Compute Causal LM Loss (Standard CrossEntropy).
-   **Generation:** Implement `generate()` method for inference.

### 2. Data (`src/data/bigrec_datamodule.py` or adapt existing)
We can reuse `SASRecDataModule` but need a **Collator** that converts item IDs to text prompts.
-   **Input:** `item_seq` (IDs).
-   **Process:** Map IDs to Titles -> Format as "Instruction + Input" string.
-   **Label:** Map Target ID to Title -> Format as "Output" string.

### 3. Experiment Script (`src/exp/run_bigrec.py`)
A new script to run the BIGRec training.
-   Load Config (`conf/teacher/bigrec.yaml`).
-   Instantiate `BigRecModel`.
-   Run `pl.Trainer`.

### 4. Configuration (`conf/teacher/bigrec.yaml`)
-   `llm_model_name`: e.g., `Qwen/Qwen2.5-0.5B-Instruct`
-   `lora_r`, `lora_alpha`, `lora_dropout`
-   `max_source_length`, `max_target_length`

## Key Challenges
-   **Tokenization Overhead:** Tokenizing text batches on the fly in the training loop might be slow. Pre-processing or efficient collators are needed.
-   **Grounding:** Mapping generated titles back to Item IDs for evaluation. We will implement a simple exact match or substring match initially.
