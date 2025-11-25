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
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_dtype = llm_dtype
        
        # LLMをMoeLoraModelでラップする
        self.llm = MoeLoraModel(
            model=llm,
            target_modules=["q_proj", "v_proj"],
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_experts=num_lora_experts,
        ).to(self.device) # LLMはfactory.pyですでに正しいdtypeにキャストされています

        # ベースLLMの全パラメータの凍結は、MoeLoraModel内で
        # 元の重みのrequires_grad=Falseを設定することで処理されます

        self.tokenizer = tokenizer
        self.rec_model = rec_model.to(self.device) # SASRecモデルの出力はilora_model.py内でキャストされます
        self.projector = projector.to(self.device) # Projectorはfactory.py内でキャストされます
        self.candidate_topk = candidate_topk
        self.item_id_to_name = item_id_to_name
        self.padding_item_id = padding_item_id

        # item_idから名前への高速なルックアップのためのリストを作成（SASRecDatasetのitem_id_to_nameのためにまだ必要）
        # item_idが0または1から始まり、連続した範囲をカバーしていると仮定
        max_item_id = max(self.item_id_to_name.keys())
        self.item_name_lookup = [None] * (max_item_id + 1)
        for item_id, item_name in self.item_id_to_name.items():
            self.item_name_lookup[item_id] = item_name

        # ゲーティングネットワークはエキスパートの重みを予測します
        self.gating_network = MLPProjector(
            input_dim=self.rec_model.hidden_size, # ゲーティングのための正しい入力次元
            output_dim=num_lora_experts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        ).to(self.device, dtype=self.llm_dtype) # ゲーティングネットワークをllm_dtypeにキャスト

        # LLMの最後の隠れ状態をnum_items個のロジットに射影する線形層を追加
        self.item_prediction_head = nn.Linear(self.llm.model.config.hidden_size, num_items).to(self.device, dtype=self.llm_dtype) # item_prediction_headをllm_dtypeにキャスト

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        """
        SASRecのアイテム表現をLLMの埋め込み空間に射影します。
        items: (batch_size, seq_len, hidden_size) の埋め込み、または (batch_size, seq_len) のID
        """
        if items.dtype == torch.long or items.dtype == torch.int:
            # IDが渡された場合、SASRecモデルを使って埋め込みを取得
            # SASRecモデルの実装に依存するが、通常はEmbedding層を持つ
            if hasattr(self.rec_model, "item_embeddings"):
                item_embs = self.rec_model.item_embeddings(items)
            elif hasattr(self.rec_model, "cacu_x"): # For SASRecModules_ori.py compatibility
                item_embs = self.rec_model.cacu_x(items)
            else:
                raise AttributeError("rec_model does not have 'item_embeddings' or 'cacu_x' method.")
            
            return self.projector(item_embs.to(self.projector.model[0].weight.dtype))
        else:
            # すでに埋め込みの場合はそのまま射影
            return self.projector(items.to(self.projector.model[0].weight.dtype))

    def encode_users(self, last_item_representation: torch.Tensor) -> torch.Tensor:
        """
        最後のアイテム表現（ユーザー埋め込み）をゲーティングに使用します。
        """
        # last_item_representationはすでにSASRecからの出力であると仮定
        return last_item_representation

    def _get_llm_outputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        入力を準備し、LLMを実行し、生の出力を返します。
        このメソッドは以下を処理します:
        1. SASRecからのユーザー埋め込みの計算。
        2. ユーザー埋め込みからのゲート重みの計算。
        3. プロンプト内の[HistoryEmb]プレースホルダーの、射影されたアイテム埋め込みへの置換。
        4. MoE-LoRA LLMの実行。
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
        self.llm.gate_weights.clear()
        self.llm.gate_weights.append(gate_weights)

        # 置換用のアイテム埋め込みの準備
        his_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[HistoryEmb]")
        ]
        his_item_embeds = self.encode_items(full_sequence_rec_embs)

        modified_input_embeds = input_embeds.clone()

        # プレースホルダー埋め込みのベクトル化された置換
        history_emb_mask = (input_ids == his_token_id)
        
        # シーケンス内の有効なアイテム（パディングでなく、seq_len内）のマスクを作成
        max_seq_len = item_seq.shape[1]
        seq_len_mask = torch.arange(max_seq_len, device=item_seq.device)[None, :] >= (max_seq_len - item_seq_len[:, None])
        valid_item_mask = seq_len_mask & (item_seq != self.padding_item_id)

        # ランク計算 (後ろから0, 1, 2...)
        # Placeholders
        p_cumsum = history_emb_mask.cumsum(dim=1)
        p_count = p_cumsum[:, -1].unsqueeze(1)
        p_rank = p_count - p_cumsum # 0-based rank from end (0 is last)

        # Valid Items
        i_cumsum = valid_item_mask.cumsum(dim=1)
        i_count = i_cumsum[:, -1].unsqueeze(1)
        i_rank = i_count - i_cumsum # 0-based rank from end (0 is last)

        # バッファへのスキャッター (Items -> Buffer)
        # バッファサイズは max_seq_len
        # batch_indicesを用意
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
        
        # アイテムをランクに基づいてバッファに配置
        # valid_item_maskがTrueの場所のみ
        item_buffer = torch.zeros(batch_size, max_seq_len, his_item_embeds.shape[-1], device=his_item_embeds.device, dtype=his_item_embeds.dtype)
        
        # Advanced indexing
        b_idx_i = batch_indices.expand_as(valid_item_mask)[valid_item_mask]
        rank_idx_i = i_rank[valid_item_mask]
        
        # ランクが負になることはないはずだが、念のためチェック (mask外は計算上どうなる？)
        # mask外はrankがcount以上になるか、負になるか。
        # mask外は無視するのでOK。
        
        # 安全のためにランクをクランプ (index out of boundsを防ぐ)
        # ただし論理的に正しいなら不要。max_seq_len内に収まるはず。
        rank_idx_i = rank_idx_i.clamp(0, max_seq_len - 1)
        
        item_buffer[b_idx_i, rank_idx_i] = his_item_embeds[valid_item_mask]

        # バッファからのギャザー (Buffer -> Placeholders)
        # history_emb_maskがTrue かつ p_rank < i_count の場所のみ置換
        replace_mask = history_emb_mask & (p_rank < i_count)
        
        b_idx_p = batch_indices.expand_as(replace_mask)[replace_mask]
        rank_idx_p = p_rank[replace_mask]
        rank_idx_p = rank_idx_p.clamp(0, max_seq_len - 1)
        
        modified_input_embeds[replace_mask] = item_buffer[b_idx_p, rank_idx_p].to(modified_input_embeds.dtype)

        # --- [CansEmb] Replacement Logic ---
        cans_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[CansEmb]")
        ]
        
        # Check if [CansEmb] exists in the batch
        if (input_ids == cans_token_id).any():
            # Encode candidate items
            # batch["cans"] should be available now
            if "cans" in batch:
                cans_item_embeds = self.encode_items(batch["cans"]) # (batch_size, num_candidates, hidden_size)
                
                cans_emb_mask = (input_ids == cans_token_id)
                
                # Rank calculation for candidates
                # Assuming candidates are already valid and ordered?
                # Usually candidates are dense (no padding in middle).
                # But we might have padding at end if num_candidates varies?
                # batch["len_cans"] exists.
                
                max_cans_len = cans_item_embeds.shape[1]
                if "len_cans" in batch:
                    cans_len_mask = torch.arange(max_cans_len, device=input_ids.device)[None, :] < batch["len_cans"][:, None]
                else:
                    cans_len_mask = torch.ones(batch_size, max_cans_len, device=input_ids.device, dtype=torch.bool)
                    
                # Placeholders ranks
                cp_cumsum = cans_emb_mask.cumsum(dim=1)
                # We want to replace FIRST N placeholders with FIRST N candidates?
                # Original logic:
                # indices_to_replace = cans_placeholder_indices[:num_to_replace_cans]
                # valid_cans_embeds = cans_item_embeds[i][:num_to_replace_cans]
                # So it's FIRST to FIRST. (0-th candidate to 0-th placeholder)
                
                # So we need 0-based rank from START.
                # cp_rank = cp_cumsum - 1. (where mask is True)
                cp_rank = cp_cumsum - 1
                
                # Candidates ranks
                # Since candidates are dense and ordered, the index IS the rank.
                # We don't need scatter/gather for candidates if we assume they are dense.
                # We can just use the index.
                
                # But we need to match Placeholder Rank K to Candidate Index K.
                
                # Create a buffer? Or just index into cans_item_embeds directly?
                # cans_item_embeds is (B, C, H).
                # We want: modified[b, p_idx] = cans_item_embeds[b, cp_rank[b, p_idx]]
                
                # Condition: mask is True AND cp_rank < num_valid_cans
                
                if "len_cans" in batch:
                    num_valid_cans = batch["len_cans"].unsqueeze(1)
                else:
                    num_valid_cans = torch.tensor(max_cans_len, device=input_ids.device).unsqueeze(0).expand(batch_size, 1)
                    
                c_replace_mask = cans_emb_mask & (cp_rank < num_valid_cans)
                
                b_idx_c = batch_indices.expand_as(c_replace_mask)[c_replace_mask]
                rank_idx_c = cp_rank[c_replace_mask]
                rank_idx_c = rank_idx_c.clamp(0, max_cans_len - 1)
                
                modified_input_embeds[c_replace_mask] = cans_item_embeds[b_idx_c, rank_idx_c].to(modified_input_embeds.dtype)
            else:
                logger.warning("Batch does not contain 'cans' key, skipping [CansEmb] replacement.")

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

    def get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        蒸留用のフォワードパス。ランキングスコア、埋め込み、候補、信頼度を返します。
        """
        outputs = self._get_llm_outputs(batch)
        
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        ranking_scores = self.item_prediction_head(last_hidden_state)
        
        confidence, candidates = torch.topk(ranking_scores, k=self.candidate_topk, dim=-1)
        confidence = F.softmax(confidence, dim=-1)

        return {
            "ranking_scores": ranking_scores,
            "embeddings": last_hidden_state,
            "candidates": candidates,
            "confidence": confidence,
        }


