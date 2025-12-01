# coding=utf-8
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.teacher.moe_lora_model import MoeLoraModel
from src.teacher.mlp_projector import MLPProjector

logger = logging.getLogger(__name__)


class iLoRAModel(nn.Module):
    """
    iLoRA (Implicit Low-Rank Adaptation) モデル。
    シーケンシャル推薦モデル (SASRec) と大規模言語モデル (LLM) を
    Mixture-of-Experts (MoE) LoRA アプローチを用いて組み合わせます。
    
    SASRecモデルはユーザー履歴をエンコードし、以下の用途に使用されます:
    1. LLM内のMoE-LoRAレイヤーに対するゲート重みの生成。
    2. LLMプロンプト内のプレースホルダーを置き換えるアイテム埋め込みの提供。
    """
    def __init__(
        self,
        llm: AutoModel,
        tokenizer: AutoTokenizer,
        num_items: int,
        num_lora_experts: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        hidden_size: int,  # ゲーティングネットワークの隠れ層サイズ
        dropout_rate: float,
        candidate_topk: int,
        item_id_to_name: Dict[int, str],
        padding_item_id: int,
        llm_dtype: torch.dtype,
        original_vocab_size: int, # アイテムIDのオフセット
        sasrec_model: Optional[nn.Module] = None, # アンサンブル用SASRecモデル
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_dtype = llm_dtype
        self.original_vocab_size = original_vocab_size
        self.num_items = num_items
        self.sasrec_model = sasrec_model
        
        # ベースLLMの全パラメータを凍結（学習させない）
        for param in llm.parameters():
            param.requires_grad = False

        # LLMをMoeLoraModelでラップする
        self.llm = MoeLoraModel(
            model=llm,
            target_modules=["q_proj", "v_proj"],
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_experts=num_lora_experts,
        ).to(self.device)

        self.tokenizer = tokenizer
        self.candidate_topk = candidate_topk
        self.item_id_to_name = item_id_to_name
        self.padding_item_id = padding_item_id

        # item_idから名前への高速なルックアップのためのリストを作成
        max_item_id = max(self.item_id_to_name.keys())
        self.item_name_lookup = [None] * (max_item_id + 1)
        for item_id, item_name in self.item_id_to_name.items():
            self.item_name_lookup[item_id] = item_name

        # ゲーティングネットワーク：ユーザー埋め込みからエキスパートの重みを予測します
        # 入力次元はLLMの隠れ層サイズ（アイテム埋め込みの平均を使うため）
        self.gating_network = MLPProjector(
            input_dim=self.llm.model.config.hidden_size,
            output_dim=num_lora_experts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        ).to(self.device, dtype=self.llm_dtype)

        self.item_prediction_head = None # Embedding Headのみ使用
        
        # アンサンブル用ゲート (SASRec Hidden -> 1)
        if self.sasrec_model is not None:
            self.ensemble_gate = nn.Linear(self.sasrec_model.hidden_size, 1).to(self.device)
            nn.init.zeros_(self.ensemble_gate.bias) # 初期値はSigmoid(0)=0.5で中立に
            print("Ensemble Gate initialized.")
            
            # SASRec Projector (SASRec Hidden -> LLM Hidden) for Hybrid Input
            self.sasrec_projector = nn.Linear(self.sasrec_model.hidden_size, self.llm.model.config.hidden_size).to(self.device)
            print("SASRec Projector initialized for Hybrid Input.")

    def _get_llm_outputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        入力を準備し、LLMを実行し、生の出力を返します。
        E4SRecアプローチ: input_idsにはすでにアイテムトークンが含まれています。
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # --- Gating Logic ---
        # 履歴アイテムの埋め込みを平均してユーザー表現とし、ゲーティングに使用する
        # input_idsの中で、アイテムトークン（>= original_vocab_size）を探す
        # ただし、最後のアイテムトークンは「ターゲット（正解）」なので、履歴には含めない
        
        # 1. アイテムトークンのマスクを作成
        is_item_token = input_ids >= self.original_vocab_size
        
        # 2. 各バッチについて、最後のアイテムトークン（ターゲット）を除外する
        # これは少し複雑だが、E4SRecのCollatorは [Prefix, Items, Suffix, Target] の順で作っている
        # ターゲットは必ず最後（またはSuffixの後）にあるはず。
        # 簡易的に、is_item_tokenがTrueになる最後のインデックスをFalseにする
        
        # バッチ処理のためにループを使わずベクトル化したいが、可変長なので難しい
        # ここではループで処理する（バッチサイズは小さいので許容範囲）
        # あるいは、Collatorで「履歴マスク」を作って渡してもらうのがベストだが、
        # ここではinput_idsから推測する。
        
        input_embeddings = self.llm.get_input_embeddings()(input_ids) # (B, Seq, H)
        
        batch_gate_inputs = []
        for i in range(input_ids.shape[0]):
            # このサンプルのアイテムトークン位置
            item_indices = torch.nonzero(is_item_token[i]).squeeze(-1)
            
            if len(item_indices) > 0:
                # すべてのアイテムトークンを履歴とする（ターゲットはinput_idsに含まれていないため）
                history_indices = item_indices
            else:
                history_indices = torch.tensor([], device=self.device, dtype=torch.long)
                
            if len(history_indices) > 0:
                # 履歴アイテムの埋め込みを平均
                history_embs = input_embeddings[i, history_indices, :]
                user_emb = history_embs.mean(dim=0)
            else:
                user_emb = torch.zeros(self.llm.model.config.hidden_size, device=self.device, dtype=self.llm_dtype)
                
            batch_gate_inputs.append(user_emb)
            
        gate_inputs = torch.stack(batch_gate_inputs) # (B, H)
        
        # ゲート重みの計算
        gate_inputs = gate_inputs.to(self.gating_network.model[0].weight.dtype)
        gate_weights = F.softmax(self.gating_network(gate_inputs), dim=-1)
        gate_weights = gate_weights.to(input_embeddings.dtype)

        # MoeLoraModelインスタンスにgate_weightsを設定
        self.llm.gate_weights.clear()
        self.llm.gate_weights.append(gate_weights)
        
        # --- Hybrid Input Injection ---
        if self.sasrec_model is not None and hasattr(self, 'sasrec_projector'):
            # input_idsの中でアイテムトークンに対応する部分に、SASRecのEmbeddingを射影して加算する
            
            # 1. アイテムIDを取得 (TokenID - Offset)
            # マスクされている部分（非アイテム）は無視
            # is_item_tokenは既に計算済み
            
            # 2. SASRec Embeddingを取得
            # SASRecのEmbedding行列: (NumItems+1, H_rec)
            # TokenIDからItemIDへの変換: token_id - original_vocab_size
            
            # ベクトル化して処理
            # input_ids全体に対して計算し、is_item_tokenでマスクして加算する
            
            # Token ID -> Item ID (1-based)
            # 非アイテムトークンは負になるが、gatherでエラーになるのでclampする
            item_ids = (input_ids - self.original_vocab_size).clamp(min=0)
            
            # SASRec Embeddings (B, Seq, H_rec)
            sasrec_embs = self.sasrec_model.item_embeddings(item_ids)
            
            # Project (B, Seq, H_llm)
            projected_sasrec_embs = self.sasrec_projector(sasrec_embs)
            
            # 加算 (In-placeだと勾配がおかしくなる可能性があるので新しいTensorを作る)
            # is_item_tokenをブロードキャスト用に拡張 (B, Seq, 1)
            mask = is_item_token.unsqueeze(-1).to(input_embeddings.dtype)
            
            # input_embeddings = input_embeddings + mask * projected_sasrec_embs
            # 元のinput_embeddingsはSemantic Initされている
            input_embeddings = input_embeddings + mask * projected_sasrec_embs.to(input_embeddings.dtype)

        # LLM実行
        # input_idsをそのまま渡す（埋め込みルックアップはLLM内部で行われる）
        # ただし、input_embeddingsを計算済みなので、inputs_embedsとして渡す方が効率的かも？
        # MoeLoraModelはinputs_embedsを受け取れるか？ -> Base Model (AutoModel) は受け取れる。
        # MoeLoraModel.forward -> self.model(...)
        
        outputs = self.llm(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return outputs

    def set_gate_weights(self, input_ids: torch.Tensor):
        """
        input_idsに基づいてゲート重みを計算し、MoeLoraModelに設定します。
        generate()を呼び出す前に使用します。
        """
        # 1. アイテムトークンのマスクを作成
        is_item_token = input_ids >= self.original_vocab_size
        
        # 2. Embedding取得 (Lookup only)
        input_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        batch_gate_inputs = []
        for i in range(input_ids.shape[0]):
            # このサンプルのアイテムトークン位置
            item_indices = torch.nonzero(is_item_token[i]).squeeze(-1)
            
            if len(item_indices) > 0:
                # すべてのアイテムトークンを履歴とする
                history_indices = item_indices
            else:
                history_indices = torch.tensor([], device=self.device, dtype=torch.long)
                
            if len(history_indices) > 0:
                # 履歴アイテムの埋め込みを平均
                history_embs = input_embeddings[i, history_indices, :]
                user_emb = history_embs.mean(dim=0)
            else:
                user_emb = torch.zeros(self.llm.model.config.hidden_size, device=self.device, dtype=self.llm_dtype)
                
            batch_gate_inputs.append(user_emb)
            
        gate_inputs = torch.stack(batch_gate_inputs) # (B, H)
        
        # ゲート重みの計算
        gate_inputs = gate_inputs.to(self.gating_network.model[0].weight.dtype)
        gate_weights = F.softmax(self.gating_network(gate_inputs), dim=-1)
        gate_weights = gate_weights.to(input_embeddings.dtype)

        # MoeLoraModelインスタンスにgate_weightsを設定
        self.llm.gate_weights.clear()
        self.llm.gate_weights.append(gate_weights)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self._get_llm_outputs(batch)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        蒸留・評価用のフォワードパス。
        """
        outputs = self._get_llm_outputs(batch)
        
        # 最後のトークンの隠れ状態を取得
        last_token_indices = batch["attention_mask"].sum(1) - 1
        last_hidden_state = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0], device=self.device), last_token_indices, :] # (B, H)
        
        # ランキングスコア（ロジット）の計算
        # LLMのアイテムEmbedding（拡張された語彙部分）との内積
        
        # 全アイテムのEmbeddingを取得 (N+1, H)
        # weight: (Vocab, H)
        # items start at original_vocab_size (ID 0 = Padding)
        all_embeddings = self.llm.get_input_embeddings().weight
        item_embeddings = all_embeddings[self.original_vocab_size : self.original_vocab_size + self.num_items + 1]
        
        # 内積 (B, H) @ (N, H).T -> (B, N)
        ranking_scores = last_hidden_state @ item_embeddings.T
        
        # パディングアイテムのスコアをマスク
        if self.padding_item_id is not None:
             ranking_scores[:, self.padding_item_id] = float('-inf')
             
        # --- Ensemble Logic ---
        alpha = None
        if self.sasrec_model is not None:
            # SASRec Forward
            # batch["seq"] contains raw item IDs for SASRec
            sasrec_ids = batch["seq"]
            sasrec_lens = batch["len_seq"] # Assuming this key exists or compute it
            if sasrec_lens is None:
                 sasrec_lens = (sasrec_ids != 0).sum(dim=1)
            
            # SASRec Logits (B, NumItems) - 0-based? No, SASRec usually outputs for all items.
            # SASRec.predict returns scores for items 1..N (padding 0 excluded usually or handled)
            # Let's assume predict returns (B, NumItems) corresponding to item IDs 1..N
            # Our ranking_scores is (B, NumItems+1) including padding at 0.
            
            sasrec_logits = self.sasrec_model.predict(sasrec_ids, sasrec_lens) # (B, NumItems)
            
            # Pad SASRec logits at index 0 to match ranking_scores shape (B, NumItems+1)
            # Or just ignore index 0 in ranking_scores.
            # Let's pad SASRec logits with -inf at index 0
            padding_scores = torch.full((sasrec_logits.size(0), 1), float('-inf'), device=self.device)
            sasrec_logits_padded = torch.cat([padding_scores, sasrec_logits], dim=1)
            
            # SASRec User Embedding for Gating
            sasrec_user_emb = self.sasrec_model(sasrec_ids, sasrec_lens) # (B, Hidden)
            
            # Compute Alpha
            alpha_logits = self.ensemble_gate(sasrec_user_emb)
            alpha = torch.sigmoid(alpha_logits) # (B, 1)
            
            # Combine Scores
            # ranking_scores is LLM scores
            ranking_scores = alpha * sasrec_logits_padded + (1 - alpha) * ranking_scores
        
        # 上位K個の候補と信頼度
        confidence, candidates = torch.topk(ranking_scores, k=self.candidate_topk, dim=-1)
        confidence = F.softmax(confidence, dim=-1)

        return {
            "ranking_scores": ranking_scores,
            "embeddings": last_hidden_state,
            "candidates": candidates,
            "confidence": confidence,
            "alpha": alpha
        }


