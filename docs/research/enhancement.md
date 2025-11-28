# Academic Enhancement Plan for Distillation Methodology

本ドキュメントでは、現在実装されている「iLoRA Teacher -> SASRec Student」の蒸留パイプラインを、学術的な新規性を主張できるレベルまで昇華させるための拡張手法を提案します。

## 1. 関連研究サーベイ (Literature Review)

本提案の立ち位置を明確にするため、関連する主要な研究分野と代表的な論文を整理します。

### 1.1. LLM for Recommendation (LLM4Rec)
LLMを推薦システムに活用する研究は急増しており、大きく「LLMを直接Recommenderとして使う」派と「LLMをデータ拡張や特徴抽出に使う」派に分かれます。
- **TALLRec (RecSys 2023)**: LLMをInstruction Tuningして推薦タスクに特化させる手法。
- **LLM-Rec (WSDM 2024)**: LLMを用いてテキスト記述の拡張や高品質なネガティブサンプリングを行う手法。
- **ONCE (2023)**: Open-source LLMと商用LLMの知識を蒸留するGenerative Recommendationの枠組み。

### 1.2. Knowledge Distillation in RecSys
推薦システムにおける蒸留は、モデル圧縮やランキング精度の向上を目的として広く研究されています。
- **RocketQA (NAACL 2021)**: 検索分野において、Cross-Encoder (Teacher) から Dual-Encoder (Student) への蒸留を行い、さらにネガティブサンプリングを強化する手法。
- **Dual-Teacher Knowledge Distillation**: 複数のTeacherからの蒸留。
- **Privileged Features Distillation**: 学習時にのみ利用可能なリッチな特徴（画像、テキスト全文など）をTeacherに入力し、Studentには基本特徴のみで推論させる手法。

### 1.3. Mutual Learning & Co-Training
TeacherとStudentが相互に高め合うアプローチです。
- **Deep Mutual Learning (CVPR 2018)**: 複数のネットワークが互いの出力を正解として学習し合う。

---

## 2. 提案手法 (Proposed Enhancements)

既存の「一方通行の蒸留」や「単純な逆蒸留」を超え、**iLoRA (MoE + LoRA)** というアーキテクチャの特性を活かした、新規性の高い3つの手法を提案します。

### 提案1: Uncertainty-Aware MoE Distillation (UAMD)
**「MoEの専門家間の意見の不一致を不確実性として利用する」**

*   **背景:** 通常の蒸留では、Teacherの出力（ソフトラベル）を無条件に正解として扱います。しかし、Teacher自身が自信がない（不確実な）ケースで無理に蒸留すると、Studentにノイズを教えることになります。
*   **手法:**
    1.  iLoRA Teacherは $N$ 個のLoRA Expertを持ちます。推論時、Gating Networkによる加重平均だけでなく、**各Expertの出力分布の分散（Variance）** を計算します。
    2.  この分散を「Teacherの認識論的不確実性 (Epistemic Uncertainty)」のプロキシとして定義します。
    3.  蒸留損失関数において、不確実性が高いサンプルの重みを動的に下げます（または、不確実性が高い場合はHard LabelであるGround Truthを優先します）。
    *   $L_{total} = (1 - \alpha) L_{CE} + \alpha \cdot w(u) \cdot L_{KD}$
    *   ここで $w(u)$ は不確実性 $u$ の減少関数です。
*   **新規性:** MoEアーキテクチャ固有の構造（Expert分散）を、追加の計算コストなしで不確実性推定に利用し、蒸留の信頼性を制御する点。

### 提案2: Rationale-Distilled Representation (RDR)
**「LLMの『推論過程』をベクトル空間に蒸留する」**

*   **背景:** LLMの強みは「なぜそのアイテムを推薦するか」という推論（Reasoning）能力にあります。しかし、SASRecのようなベクトルモデルは結果（ID）しか出力できません。
*   **手法:**
    1.  **Rationale Generation:** Teacher (LLM) に、推薦アイテムだけでなく「推薦理由（Rationale）」も生成させます（例：「ユーザーはSF映画を好んでおり、特にタイムトラベルものが好きだから」）。
    2.  **Rationale Encoding:** 生成されたRationaleを、TeacherのEmbedding Head（または別のText Encoder）でベクトル化します ($v_{reason}$)。
    3.  **Representation Alignment:** Studentの中間層出力（または最終アイテム表現）が、この $v_{reason}$ と類似するように正則化項を追加します。
    *   $L_{reason} = || E_{student}(user\_hist) - v_{reason} ||^2$
*   **新規性:** 単なる出力確率（Logits）の蒸留ではなく、LLMの言語化能力（Reasoning）を潜在空間の制約としてStudentに注入する点。「Explainable Recommendation」の能力を間接的にStudentに持たせる試み。

### 提案3: Cyclic Dual-Refinement (CDR)
**「StudentとTeacherの役割を動的に入れ替える反復的共進化」**

*   **背景:** 現在の実装（Reverse Distillation）は、Student -> Teacherへの正則化という静的なものです。これを動的なプロセスにします。
*   **手法:** 以下の3ステップを1ラウンドとし、複数ラウンド繰り返します。
    1.  **Step 1 (Student Pre-training):** 通常のSASRec学習。
    2.  **Step 2 (Teacher Refinement):** TeacherのItem EmbeddingをStudentのもので初期化し、LLM部分を固定してItem EmbeddingのみをFine-tuning（Reverse Distillation付き）。ここでTeacherは「Studentの知識」＋「LLMの推論」を融合した最強のEmbeddingを獲得します。
    3.  **Step 3 (Student Distillation):** RefineされたTeacherを用いてStudentを蒸留。ここでStudentはTeacherの高度な知識を吸収します。
    4.  **Loop:** Step 3で賢くなったStudentのEmbeddingを、次のラウンドのStep 2の初期値として利用します。
*   **新規性:** 推薦システムにおけるLLMとIDベースモデルの「共進化（Co-evolution）」フレームワークとして定式化する点。特に、Item Embeddingを共通言語（Interface）としてループさせる点がユニークです。

---

## 3. 実験計画 (Experimental Plan)

各提案手法の有効性を検証するための実験設定です。

| 手法 | 比較対象 (Baseline) | 評価指標 | 期待される結果 |
| :--- | :--- | :--- | :--- |
| **UAMD** | 通常のiLoRA蒸留, 一様重み付け | NDCG@10, MRR, Calibration Error | Teacherが苦手なロングテールアイテム等でのStudentの精度低下を防げる。 |
| **RDR** | 通常のiLoRA蒸留 | NDCG@10, 説明性の定性評価 (t-SNE等) | 推薦理由に基づいたクラスタリングが形成され、納得感のある推薦が増える。 |
| **CDR** | 単発の蒸留 (One-pass) | NDCG@10 (Roundごとの推移) | ラウンドを重ねるごとに精度が向上し、ある時点で飽和する（収束する）。 |

## 4. 結論

これらの手法は、単に「LLMを使った」だけでなく、**「LLMの構造（MoE）」「LLMの能力（Reasoning）」「LLMとIDモデルの相互作用（Cyclic）」** に着目しており、トップカンファレンス（RecSys, SIGIR, WWW等）でも十分に通じる新規性を持っています。まずは実装コストが比較的低い **提案1 (UAMD)** または **提案3 (CDR)** から着手することを推奨します。
