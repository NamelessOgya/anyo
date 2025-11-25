import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.teacher.moe_lora_model import MoeLoraModel
from src.teacher.mlp_projector import MLPProjector

logger = logging.getLogger(__name__)


class iLoRAModel(nn.Module):
    def __init__(
        self,
        llm: AutoModelForCausalLM,
        tokenizer: AutoTokenizer, # Re-added
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
        llm_dtype: torch.dtype, # Add new argument
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_dtype = llm_dtype # Store llm_dtype
        
        # Wrap the LLM with MoeLoraModel
        self.llm = MoeLoraModel(
            model=llm,
            target_modules=["q_proj", "v_proj"],
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_lora_experts=num_lora_experts,
        ).to(self.device) # LLM is already cast to correct dtype in factory.py

        # Freeze all parameters of the base LLM is handled within MoeLoraModel
        # by setting requires_grad=False on original weights

        self.tokenizer = tokenizer # Re-added
        self.rec_model = rec_model.to(self.device) # SASRec model's outputs are cast in ilora_model.py
        self.projector = projector.to(self.device) # Projector is cast in factory.py
        self.candidate_topk = candidate_topk
        self.item_id_to_name = item_id_to_name
        self.padding_item_id = padding_item_id

        # Build a list for faster item_id to name lookup (still needed for SASRecDataset's item_id_to_name)
        # Assuming item_id starts from 0 or 1 and covers a contiguous range
        max_item_id = max(self.item_id_to_name.keys())
        self.item_name_lookup = [None] * (max_item_id + 1)
        for item_id, item_name in self.item_id_to_name.items():
            self.item_name_lookup[item_id] = item_name

        # The gating network will predict expert weights
        self.gating_network = MLPProjector(
            input_dim=self.rec_model.hidden_size, # Correct input dim for gating
            output_dim=num_lora_experts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        ).to(self.device, dtype=self.llm_dtype) # Cast gating network to llm_dtype

        # Add a linear layer to project LLM's last hidden state to num_items logits
        self.item_prediction_head = nn.Linear(self.llm.model.config.hidden_size, num_items).to(self.device, dtype=self.llm_dtype) # Cast item_prediction_head to llm_dtype

    # _prepare_llm_input method removed
    # def _prepare_llm_input(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, torch.Tensor]:
    #    ... (removed content) ...

    def encode_items(self, full_sequence_representations: torch.Tensor) -> torch.Tensor:
        # Assumes full_sequence_representations is already output from SASRec's Transformer blocks
        # No need to call rec_model again
        return self.projector(full_sequence_representations.to(self.projector.model[0].weight.dtype))

    def encode_users(self, last_item_representation: torch.Tensor) -> torch.Tensor:
        # Assumes last_item_representation is already output from SASRec
        # No need to call rec_model again
        return last_item_representation

    def _get_llm_outputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Prepares inputs, runs the LLM, and returns its raw outputs.
        This is a common private method for `forward` and `get_teacher_outputs`.
        """
        # DEBUG PRINT AT THE VERY BEGINNING
        # logger.info(f"DEBUG(iLoRA._get_llm_outputs): Input batch['input_ids'][0]: {batch['input_ids'][0]}")
        # logger.info(f"DEBUG(iLoRA._get_llm_outputs): Input batch['input_ids'][0] tokens: {self.tokenizer.convert_ids_to_tokens(batch['input_ids'][0].tolist())}")
        # END DEBUG PRINT

        item_seq, item_seq_len = batch["seq"], batch["len_seq"]

        # input_ids and attention_mask are now directly from the batch, prepared by collate_fn
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        input_embeds = self.llm.get_input_embeddings()(input_ids) # Define input_embeds early

        # Get both full sequence representations and last item representation from SASRec in one pass
        full_sequence_rec_embs = self.rec_model.get_full_sequence_representations(item_seq, item_seq_len)
        last_item_rec_representation = self.rec_model._get_last_item_representation(item_seq, item_seq_len)

        user_embeds = self.encode_users(last_item_rec_representation)
        user_embeds = user_embeds.to(self.gating_network.model[0].weight.dtype) # Cast to match gating_network dtype
        gate_weights = F.softmax(self.gating_network(user_embeds), dim=-1)
        gate_weights = gate_weights.to(input_embeds.dtype) # input_embeds.dtype is now available

        # Set gate_weights directly on the MoeLoraModel instance
        self.llm.gate_weights.clear()
        self.llm.gate_weights.append(gate_weights)

        his_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[HistoryEmb]")
        ]
        his_item_embeds = self.encode_items(full_sequence_rec_embs) # Pass full sequence representations here

        modified_input_embeds = input_embeds.clone()

        # Vectorized replacement of placeholder embeddings
        history_emb_mask = (input_ids == his_token_id)
        
        # DEBUG PRINTS START
        # logger.info(f"DEBUG(iLoRA._get_llm_outputs): history_emb_mask sum (after calculation): {history_emb_mask.sum()}")
        # logger.info(f"DEBUG(iLoRA._get_llm_outputs): input_ids dtype: {input_ids.dtype}")
        # logger.info(f"DEBUG(iLoRA._get_llm_outputs): his_token_id type: {type(his_token_id)}")
        # DEBUG PRINTS END

        # Create a mask for valid items in the sequence (not padding and within seq_len)
        max_seq_len = item_seq.shape[1]
        seq_len_mask = torch.arange(max_seq_len, device=item_seq.device)[None, :] >= (max_seq_len - item_seq_len[:, None])
        valid_item_mask = seq_len_mask & (item_seq != self.padding_item_id)
        
        # Loop over each sample in the batch to handle potential mismatches from truncation
        for i in range(input_ids.shape[0]):
            sample_placeholder_mask = history_emb_mask[i]
            sample_valid_item_mask = valid_item_mask[i]
            
            num_placeholders = sample_placeholder_mask.sum()
            num_valid_items = sample_valid_item_mask.sum()

            if num_placeholders == 0:
                continue

            # Determine the number of embeddings to replace
            num_to_replace = min(num_placeholders, num_valid_items).item()

            if num_to_replace > 0:
                # Get the last `num_to_replace` valid item embeddings
                valid_item_embeds = his_item_embeds[i][sample_valid_item_mask]
                last_valid_item_embeds = valid_item_embeds[-num_to_replace:]

                # Get the indices of all placeholders for the current sample
                placeholder_indices = torch.where(sample_placeholder_mask)[0]
                # Get the indices of the placeholders to be replaced (can be first or last, let's use last)
                last_placeholder_indices_to_replace = placeholder_indices[-num_to_replace:]
                
                # Create a new, specific mask for the placeholders we are actually going to replace
                final_placeholder_mask = torch.zeros_like(sample_placeholder_mask, dtype=torch.bool)
                if last_placeholder_indices_to_replace.numel() > 0:
                    final_placeholder_mask[last_placeholder_indices_to_replace] = True

                    # Perform the replacement
                    modified_input_embeds[i][final_placeholder_mask] = last_valid_item_embeds.to(modified_input_embeds.dtype)
        
            if num_placeholders != num_valid_items:
                logger.warning(
                    f"Sample {i}: Mismatch between placeholders ({num_placeholders}) and valid items ({num_valid_items}). "
                    f"Replaced the last {num_to_replace} placeholders with the most recent item embeddings."
                )

        outputs = self.llm(
            inputs_embeds=modified_input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return outputs

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # No longer needs item_seq, item_seq_len as separate args
        # These are in the batch dict, and input_ids/attention_mask are directly used
        return self._get_llm_outputs(batch)

    def get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # No longer needs item_seq, item_seq_len as separate args
        # These are in the batch dict, and input_ids/attention_mask are directly used
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


