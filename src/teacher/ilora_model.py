# coding=utf-8
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        llm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        num_items: int,
        num_lora_experts: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        hidden_size: int,  # ゲーティングネットワークの隠れ層サイズ
        dropout_rate: float,
        rec_model: nn.Module,
        projector: MLPProjector,
        candidate_topk: int,
        item_id_to_name: Dict[int, str],
        padding_item_id: int,
        llm_dtype: torch.dtype,
        use_item_embeddings_head: bool = True,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_dtype = llm_dtype
        self.use_item_embeddings_head = use_item_embeddings_head
        
        # ベースLLMの全パラメータを凍結（学習させない）
        for param in llm.parameters():
            param.requires_grad = False

        # LLMをMoeLoraModelでラップする
        # これにより、LoRAアダプターとMoEゲーティング機構が追加されます
        self.llm = MoeLoraModel(
            model=llm,
            target_modules=["q_proj", "v_proj"],
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_experts=num_lora_experts,
        ).to(self.device) # LLMはfactory.pyですでに正しいdtypeにキャストされています

        # ベースLLMのパラメータ凍結は、MoeLoraModel内で
        # 元の重みのrequires_grad=Falseを設定することで処理されます

        self.tokenizer = tokenizer
        self.rec_model = rec_model.to(self.device) # SASRecモデルの出力はilora_model.py内でキャストされます
        self.projector = projector.to(self.device) # Projectorはfactory.py内でキャストされます
        self.candidate_topk = candidate_topk
        self.item_id_to_name = item_id_to_name
        self.padding_item_id = padding_item_id

        # item_idから名前への高速なルックアップのためのリストを作成
        # （SASRecDatasetのitem_id_to_nameのためにまだ必要）
        # item_idが0または1から始まり、連続した範囲をカバーしていると仮定しています
        max_item_id = max(self.item_id_to_name.keys())
        self.item_name_lookup = [None] * (max_item_id + 1)
        for item_id, item_name in self.item_id_to_name.items():
            self.item_name_lookup[item_id] = item_name

        # ゲーティングネットワーク：ユーザー埋め込みからエキスパートの重みを予測します
        self.gating_network = MLPProjector(
            input_dim=self.rec_model.hidden_size, # ゲーティングのための正しい入力次元
            output_dim=num_lora_experts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        ).to(self.device, dtype=self.llm_dtype) # ゲーティングネットワークをllm_dtypeにキャスト

        # アイテム予測ヘッドの初期化
        # use_item_embeddings_headがFalseの場合、線形層を使用してロジットを計算します
        if not self.use_item_embeddings_head:
            self.item_prediction_head = nn.Linear(self.llm.model.config.hidden_size, num_items).to(self.device, dtype=self.llm_dtype)
        else:
            self.item_prediction_head = None

        # Reverse Distillation Lossのために、初期状態のアイテム埋め込みを保存します
        # Embedding Headを使用する場合（つまりアイテム埋め込みを学習する場合）にのみ必要です
        # 元の値を保持するために重みを複製（clone）してバッファに登録します
        if self.use_item_embeddings_head:
            if hasattr(self.rec_model, "item_embeddings"):
                self.register_buffer("student_item_embeddings", self.rec_model.item_embeddings.weight.clone().detach())
            elif hasattr(self.rec_model, "cacu_x"):
                 # SASRecModules_ori.py との互換性のため
                 if hasattr(self.rec_model, "item_embeddings"):
                     self.register_buffer("student_item_embeddings", self.rec_model.item_embeddings.weight.clone().detach())
                 else:
                     logger.warning("Reverse Distillation用のitem_embeddingsが見つかりませんでした。")
            else:
                logger.warning("Reverse Distillation用のitem_embeddingsが見つかりませんでした。")

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        """
        SASRecのアイテム表現をLLMの埋め込み空間に射影します。
        
        Args:
            items: (batch_size, seq_len, hidden_size) の埋め込みテンソル、
                   または (batch_size, seq_len) のアイテムIDテンソル
        
        Returns:
            torch.Tensor: LLMの次元に射影されたアイテム埋め込み
        """
        if items.dtype == torch.long or items.dtype == torch.int:
            # IDが渡された場合、SASRecモデルを使って埋め込みを取得
            if hasattr(self.rec_model, "item_embeddings"):
                item_embs = self.rec_model.item_embeddings(items)
            elif hasattr(self.rec_model, "cacu_x"): # SASRecModules_ori.py 互換性
                item_embs = self.rec_model.cacu_x(items)
            else:
                raise AttributeError("rec_model does not have 'item_embeddings' or 'cacu_x' method.")
            
            return self.projector(item_embs.to(self.projector.model[0].weight.dtype))
        else:
            # すでに埋め込みの場合はそのまま射影
            return self.projector(items.to(self.projector.model[0].weight.dtype))

    def encode_users(self, last_item_representation: torch.Tensor) -> torch.Tensor:
        """
        最後のアイテム表現（ユーザー埋め込み）をゲーティングに使用するために処理します。
        現在はそのまま返していますが、将来的な拡張のためにメソッド化しています。
        """
        # last_item_representationはすでにSASRecからの出力であると仮定
        return last_item_representation

    def _get_llm_outputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        入力を準備し、LLMを実行し、生の出力を返します。
        
        このメソッドの主な処理:
        1. SASRecからユーザー埋め込み（全シーケンス表現）を取得。
        2. ユーザー埋め込みに基づいて、MoE-LoRAのゲート重みを計算。
        3. プロンプト内の特別なプレースホルダー（[HistoryEmb], [CansEmb]）を、
           射影されたアイテム埋め込みに置換。
        4. MoE-LoRA LLMを実行。
        """
        item_seq, item_seq_len = batch["seq"], batch["len_seq"]

        # input_idsとattention_maskは、collate_fnによって準備されたバッチから直接取得されます
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        input_embeds = self.llm.get_input_embeddings()(input_ids)

        # SASRecから完全なシーケンス表現と最後のアイテム表現を1回のパスで取得
        full_sequence_rec_embs = self.rec_model.get_full_sequence_representations(item_seq, item_seq_len)
        last_item_rec_representation = self.rec_model._get_last_item_representation(item_seq, item_seq_len)

        # ゲート重みの計算
        user_embeds = self.encode_users(last_item_rec_representation)
        user_embeds = user_embeds.to(self.gating_network.model[0].weight.dtype)
        gate_weights = F.softmax(self.gating_network(user_embeds), dim=-1)
        gate_weights = gate_weights.to(input_embeds.dtype)

        # MoeLoraModelインスタンスにgate_weightsを直接設定
        # これにより、LLMの各層でこの重みが使用されます
        self.llm.gate_weights.clear()
        self.llm.gate_weights.append(gate_weights)

        # --- [HistoryEmb] の置換ロジック ---
        # プロンプト内の [HistoryEmb] トークンを、ユーザーの履歴アイテム埋め込みに置き換えます
        
        his_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[HistoryEmb]")
        ]
        his_item_embeds = self.encode_items(full_sequence_rec_embs)

        modified_input_embeds = input_embeds.clone()

        # プレースホルダー埋め込みのベクトル化された置換処理
        history_emb_mask = (input_ids == his_token_id)
        
        # シーケンス内の有効なアイテム（パディングでなく、seq_len内）のマスクを作成
        max_seq_len = item_seq.shape[1]
        seq_len_mask = torch.arange(max_seq_len, device=item_seq.device)[None, :] >= (max_seq_len - item_seq_len[:, None])
        valid_item_mask = seq_len_mask & (item_seq != self.padding_item_id)

        # ランク計算 (後ろから0, 1, 2...と数える)
        # これにより、最新のアイテムから順にプレースホルダーに埋めていくことができます
        
        # Placeholdersのランク
        p_cumsum = history_emb_mask.cumsum(dim=1)
        p_count = p_cumsum[:, -1].unsqueeze(1)
        p_rank = p_count - p_cumsum # 0-based rank from end (0 is last)

        # 有効なアイテムのランク
        i_cumsum = valid_item_mask.cumsum(dim=1)
        i_count = i_cumsum[:, -1].unsqueeze(1)
        i_rank = i_count - i_cumsum # 0-based rank from end (0 is last)

        # バッファへのスキャッター (Items -> Buffer)
        # 一度バッファに集めることで、プレースホルダーとアイテムの位置がずれていても対応可能にします
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        
        # アイテムをランクに基づいてバッファに配置
        item_buffer = torch.zeros(batch_size, max_seq_len, his_item_embeds.shape[-1], device=his_item_embeds.device, dtype=his_item_embeds.dtype)
        
        # Advanced indexingを使用して配置
        b_idx_i = batch_indices.expand_as(valid_item_mask)[valid_item_mask]
        rank_idx_i = i_rank[valid_item_mask]
        
        # 安全のためにランクをクランプ (index out of boundsを防ぐ)
        rank_idx_i = rank_idx_i.clamp(0, max_seq_len - 1)
        
        item_buffer[b_idx_i, rank_idx_i] = his_item_embeds[valid_item_mask]

        # バッファからのギャザー (Buffer -> Placeholders)
        # history_emb_maskがTrue かつ p_rank < i_count (アイテムが存在する) 場所のみ置換
        replace_mask = history_emb_mask & (p_rank < i_count)
        
        b_idx_p = batch_indices.expand_as(replace_mask)[replace_mask]
        rank_idx_p = p_rank[replace_mask]
        rank_idx_p = rank_idx_p.clamp(0, max_seq_len - 1)
        
        modified_input_embeds[replace_mask] = item_buffer[b_idx_p, rank_idx_p].to(modified_input_embeds.dtype)

        # --- [CansEmb] の置換ロジック ---
        # 候補アイテムの埋め込み置換ロジックは削除されました。
        # Dense Retrievalアプローチでは、プロンプトに候補を含める必要がないためです。

        outputs = self.llm(
            inputs_embeds=modified_input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        学習用のフォワードパス。生のLLM出力を返します。
        """
        return self._get_llm_outputs(batch)

    def train(self, mode: bool = True):
        """
        trainモードの上書き。
        rec_model (SASRec) は常にevalモード（Dropout無効化など）に保ちつつ、
        item_embeddings の学習（requires_grad=True）は許可します。
        """
        super().train(mode)
        if mode:
            # rec_modelは基本的にevalモードで動作させたい（Dropoutなどを無効化するため）
            # しかし、item_embeddingsの勾配計算はrequires_grad=Trueであれば行われる
            self.rec_model.eval()
        return self

    def get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        蒸留用のフォワードパス。
        ランキングスコア、埋め込み、候補、信頼度を含む辞書を返します。
        """
        outputs = self._get_llm_outputs(batch)
        
        # LLMの出力: (Batch, Seq, Hidden)
        # 最後のトークンの隠れ状態を取得
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] # (Batch, LLM_Hidden)
        
        # ランキングスコア（ロジット）の計算
        if self.use_item_embeddings_head:
            # Embedding Head ロジック:
            # Studentのアイテム埋め込みを正解（ターゲット）として使用します。
            # 1. 全アイテムの埋め込みを取得 (N, H_rec)
            all_items_indices = torch.arange(self.rec_model.item_embeddings.num_embeddings, device=self.device)
            all_item_embs_rec = self.rec_model.item_embeddings(all_items_indices)
            
            # 2. LLMの次元に射影 (N, H_llm)
            all_item_embs_llm = self.projector(all_item_embs_rec.to(self.projector.model[0].weight.dtype))
            
            # 3. 内積によりスコア計算 (B, N)
            ranking_scores = last_hidden_state @ all_item_embs_llm.T
        else:
            # Linear Head ロジック:
            # 従来の線形層による予測
            ranking_scores = self.item_prediction_head(last_hidden_state)
        
        # パディングアイテム（通常ID 0）のスコアをマスク
        if self.padding_item_id is not None:
             ranking_scores[:, self.padding_item_id] = float('-inf')
        
        # 上位K個の候補と信頼度（Softmax確率）を計算
        confidence, candidates = torch.topk(ranking_scores, k=self.candidate_topk, dim=-1)
        confidence = F.softmax(confidence, dim=-1)

        return {
            "ranking_scores": ranking_scores,
            "embeddings": last_hidden_state,
            "candidates": candidates,
            "confidence": confidence,
        }


