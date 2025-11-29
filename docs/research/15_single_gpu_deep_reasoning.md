# Single GPUで「深い推論」を実現する戦略

ユーザー様のご要望である「GPU一枚で、深い推論結果（Teacherの知識）を使用したモデルを作る」ための具体的な戦略を提案します。

## 結論：QLoRAを使えば可能です

**QLoRA (Quantized LoRA)** という技術を使えば、7BクラスのLLMを **4-bit量子化** してロードでき、VRAM消費を劇的に（約1/4に）削減できます。
これにより、**RTX 3090/4090 (24GB) 1枚**、あるいは **T4/A10 (16GB/24GB)** でもTeacherモデルを動かすことが可能になります。

## 具体的なアプローチ

### 1. Teacherの軽量化 (QLoRA)

現在の `anyo` は `MoE-LoRA` を使用していますが、これがQLoRAと互換性がありません（コード内で無効化されています）。
シングルGPUで動かすための現実的な解は以下の2つです。

*   **プランA（推奨）: Standard LoRA + QLoRA**
    *   MoE（Mixture-of-Experts）を諦め、E4SRecと同じ「標準的なLoRA」構成に戻します。
    *   これなら `bitsandbytes` と `peft` ライブラリを使って、**コードをほぼ書かずに即座に4-bit化** できます。
    *   **メリット:** 実装コストが低い。確実に動く。
    *   **デメリット:** MoEによるパーソナライズ能力は失われる（ただし、Teacherとしての「推論力」は維持される）。

*   **プランB（高難度）: MoE-LoRA + QLoRA**
    *   現在の `MoeLoraModel` を改造し、4-bit Linear層 (`bnb.nn.Linear4bit`) に対応させます。
    *   **メリット:** `anyo` の独自性（MoE）を維持できる。
    *   **デメリット:** 実装難易度が高い（量子化された重みとMoE計算の整合性を取る必要がある）。

### 2. Deep Reasoningの蒸留 (Distillation)

GPUリソースが限られている場合、学習プロセスを工夫します。

1.  **Offline Inference (推論のみ先行):**
    *   まずTeacherモデル（QLoRA 7B）だけを動かし、全学習データに対して「推論結果（EmbeddingやSoft Label）」を生成し、ファイルに保存します。
    *   これなら学習時のメモリを気にする必要がありません。
2.  **Student Training (学習のみ):**
    *   Teacherをメモリからアンロードし、保存したファイルを使ってStudent (SASRec) を学習させます。
    *   これならStudentの学習は非常に軽量です。

## 提案するロードマップ

まず **「プランA (Standard LoRA + QLoRA)」** で実装を進めることを強く推奨します。
「深い推論結果を使うことによる精度向上」を確認するのが先決であり、MoEはその後の最適化要素だからです。

### 必要な変更点
1.  `conf/teacher/model_type` に `lora` (非MoE) を追加。
2.  `factory.py` で `use_qlora` を有効化し、`peft` の標準 `get_peft_model` を使用するように分岐を追加。
