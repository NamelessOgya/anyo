import re
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.teacher.mlp_projector import MLPProjector
from src.teacher.moe_lora_model import MoeLoraConfig, Linear


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
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = llm.to(self.device)
        # Freeze all parameters of the base LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer
        self.rec_model = rec_model.to(self.device)
        self.projector = projector.to(self.device)
        self.gate_weights = None
        self.candidate_topk = candidate_topk

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
        ).to(self.device)
        self.output_layer = nn.Linear(self.llm.config.hidden_size, num_items + 1).to(self.device)

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
        input_ids = batch["tokens"].input_ids.squeeze(1)
        attention_mask = batch["tokens"].attention_mask.squeeze(1)
        item_seq, item_seq_len = batch["seq"], batch["len_seq"]
        batch_size = item_seq.shape[0]

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
        return self.output_layer(last_hidden_state)

    def get_teacher_outputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch["tokens"].input_ids.squeeze(1)
        attention_mask = batch["tokens"].attention_mask.squeeze(1)
        item_seq, item_seq_len = batch["seq"], batch["len_seq"]
        batch_size = item_seq.shape[0]

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
        ranking_scores = self.output_layer(last_hidden_state)
        
        # Get top-k candidates and their confidence
        confidence, candidates = torch.topk(ranking_scores, k=self.candidate_topk, dim=-1)
        confidence = F.softmax(confidence, dim=-1)

        return {
            "ranking_scores": ranking_scores,
            "embeddings": last_hidden_state,
            "candidates": candidates,
            "confidence": confidence,
        }
