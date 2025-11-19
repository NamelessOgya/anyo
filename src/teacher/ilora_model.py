import logging
import re
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.teacher.mlp_projector import MLPProjector
from src.teacher.moe_lora_model import MoeLoraConfig, Linear

logger = logging.getLogger(__name__) # Added module-level logger


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _replace_module(parent_module, child_name, new_module, old_module, ilora_model):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias"):
        if old_module.bias is not None:
            new_module.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)
        if "gating" in name:
            module.to(old_module.weight.device)
    
    # Wrap forward method
    new_module.forward = lambda x: Linear.forward(new_module, x, gate_weights=ilora_model.gate_weights)


class iLoRAModel(nn.Module):
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
        rec_model: nn.Module,  # Added rec_model
        projector: MLPProjector,  # Added projector
        candidate_topk: int, # Added candidate_topk
        item_id_to_name: Dict[int, str], # Added item_id_to_name
        padding_item_id: int, # Added padding_item_id
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm.to(self.device) # Removed .half()
        # Freeze all parameters of the base LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Enable gradient checkpointing for the LLM to save memory
        # self.llm.gradient_checkpointing_enable()

        self.tokenizer = tokenizer
        self.rec_model = rec_model.to(self.device)
        self.projector = projector.to(self.device) # Removed .half()
        self.gate_weights = None
        self.candidate_topk = candidate_topk
        self.item_id_to_name = item_id_to_name # Store item_id_to_name
        self.padding_item_id = padding_item_id # Store padding_item_id

        self.moe_lora_config = MoeLoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            num_moe=num_lora_experts,
            gating="MLP",
        )
        self._find_and_replace()

        self.gating_network = MLPProjector(
            input_dim=self.llm.config.hidden_size,
            output_dim=num_lora_experts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
        ).to(self.device) # Removed .half()

        # Add a linear layer to project LLM's last hidden state to num_items logits
        self.item_prediction_head = nn.Linear(self.llm.config.hidden_size, num_items).to(self.device) # Removed .half()

    def _prepare_llm_input(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepares LLM input (input_ids, attention_mask) from item sequences.
        Converts item IDs to names, constructs a prompt, and tokenizes it.
        """
        batch_size = item_seq.shape[0]
        prompts = []
        for i in range(batch_size):
            seq_len = item_seq_len[i].item()
            # Filter out padding_item_id and convert to original item IDs for mapping
            history_item_ids = [
                self.item_id_to_name[item_id.item()]
                for item_id in item_seq[i, :seq_len]
                if item_id.item() != self.padding_item_id
            ]
            
            # Construct prompt: "User history: item_name1, item_name2, ... [HistoryEmb] Next item:"
            # The [HistoryEmb] token will be replaced by actual embeddings later
            history_str = ", ".join(map(str, history_item_ids))
            prompt = f"User history: {history_str} [HistoryEmb] Next item:"
            prompts.append(prompt)

        # Tokenize the prompts
        tokenized_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128, # Use a fixed reasonable max_length to prevent OverflowError
        ).to(self.device)
        
        return tokenized_inputs

    def _find_and_replace(self):
        key_list = [key for key, _ in self.llm.named_modules()]
        for key in key_list:
            if any(key.endswith(target_key) for target_key in self.moe_lora_config.target_modules):
                parent, target, target_name = _get_submodules(self.llm, key)
                if isinstance(target, nn.Linear):
                    in_features, out_features = target.in_features, target.out_features
                    new_module = Linear(
                        "default",
                        in_features,
                        out_features,
                        r=self.moe_lora_config.r,
                        num_moe=self.moe_lora_config.num_moe,
                        gating=self.moe_lora_config.gating,
                        lora_alpha=self.moe_lora_config.lora_alpha,
                        lora_dropout=self.moe_lora_config.lora_dropout,
                    )
                    _replace_module(parent, target_name, new_module, target, self)

    def encode_items(self, seq: torch.Tensor) -> torch.Tensor:
        if hasattr(self.rec_model, "cacu_x"):
            item_rec_embs = self.rec_model.cacu_x(seq)
        elif hasattr(self.rec_model, "item_embeddings"):
            item_rec_embs = self.rec_model.item_embeddings(seq)
        else:
            raise NotImplementedError("rec_model does not have cacu_x or item_embeddings method.")
        return self.projector(item_rec_embs)

    def encode_users(self, seq: torch.Tensor, len_seq: torch.Tensor) -> torch.Tensor:
        if hasattr(self.rec_model, "_get_last_item_representation"):
            user_rec_embs = self.rec_model._get_last_item_representation(seq, len_seq)
        elif hasattr(self.rec_model, "cacul_h"):
            user_rec_embs = self.rec_model.cacul_h(seq, len_seq)
        elif hasattr(self.rec_model, "item_embeddings"):
            user_rec_embs = self.rec_model.item_embeddings(seq).mean(dim=1) # Use mean as a fallback
        else:
            raise NotImplementedError("rec_model does not have _get_last_item_representation, cacul_h or item_embeddings method.")
        return self.projector(user_rec_embs)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        item_seq, item_seq_len = batch["seq"], batch["len_seq"]
        batch_size = item_seq.shape[0]

        # Prepare LLM input from item sequences
        tokenized_inputs = self._prepare_llm_input(item_seq, item_seq_len)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        user_embeds = self.encode_users(item_seq, item_seq_len)
        self.gate_weights = F.softmax(self.gating_network(user_embeds), dim=-1)

        input_embeds = self.llm.get_input_embeddings()(input_ids)
        logger.info(f"Input Embeds (before replacement) Stats: "
                    f"min={input_embeds.min()}, max={input_embeds.max()}, mean={input_embeds.mean()}")
        his_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[HistoryEmb]")
        ]
        his_item_embeds = self.encode_items(batch["seq"])
        logger.info(f"His Item Embeds (after projection) Stats: "
                    f"min={his_item_embeds.min()}, max={his_item_embeds.max()}, mean={his_item_embeds.mean()}")

        for i in range(batch_size):
            if (input_ids[i] == his_token_id).nonzero(as_tuple=True)[0].shape[0] > 0:
                idx_tensor = (input_ids[i] == his_token_id).nonzero(as_tuple=True)[0]
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, : item_seq_len[i].item()]):
                    input_embeds[i, idx] = item_emb
        logger.info(f"Input Embeds (after replacement) Stats: "
                    f"min={input_embeds.min()}, max={input_embeds.max()}, mean={input_embeds.mean()}")

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # labels=labels, # Removed: labels are handled by trainer
            output_hidden_states=True,
        )
        return outputs

    def get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        item_seq, item_seq_len = batch["seq"], batch["len_seq"]
        batch_size = item_seq.shape[0]

        # Prepare LLM input from item sequences
        tokenized_inputs = self._prepare_llm_input(item_seq, item_seq_len)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        user_embeds = self.encode_users(item_seq, item_seq_len)
        self.gate_weights = F.softmax(self.gating_network(user_embeds), dim=-1)

        input_embeds = self.llm.get_input_embeddings()(input_ids)
        his_token_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("[HistoryEmb]")
        ]
        his_item_embeds = self.encode_items(batch["seq"])

        for i in range(batch_size):
            if (input_ids[i] == his_token_id).nonzero(as_tuple=True)[0].shape[0] > 0:
                idx_tensor = (input_ids[i] == his_token_id).nonzero(as_tuple=True)[0]
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, : item_seq_len[i].item()]):
                    input_embeds[i, idx] = item_emb

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        ranking_scores = self.item_prediction_head(last_hidden_state) # Project to num_items
        
        # Get top-k candidates and their confidence
        confidence, candidates = torch.topk(ranking_scores, k=self.candidate_topk, dim=-1)
        confidence = F.softmax(confidence, dim=-1)

        return {
            "ranking_scores": ranking_scores,
            "embeddings": last_hidden_state,
            "candidates": candidates,
            "confidence": confidence,
        }
