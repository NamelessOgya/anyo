# coding=utf-8
import math
import warnings
import re
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target = model.get_submodule(key)
    return parent, target, key.split(".")[-1]

class MoeLoraModel(torch.nn.Module):
    """
    A wrapper class to modify a transformer model to use MoeLora layers.
    This class handles the replacement of target modules and the passing of
    gate_weights to the custom MoeLora layers during the forward pass.
    """
    def __init__(self, model, target_modules, lora_r, lora_alpha, lora_dropout, num_lora_experts):
        super().__init__()
        self.model = model
        self.gate_weights = [] # This list will be used to pass gate_weights to the layers
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_lora_experts = num_lora_experts
        self.target_modules = target_modules

        self._find_and_replace()

    def _find_and_replace(self):
        """
        Recursively finds and replaces nn.Linear layers with MoeLoraLinear layers.
        """
        for name, module in self.model.named_modules():
            if any(target_key in name for target_key in self.target_modules):
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = self.model.get_submodule(parent_name)

                if isinstance(module, nn.Linear):
                    new_module = Linear(
                        adapter_name="default",
                        in_features=module.in_features,
                        out_features=module.out_features,
                        r=self.lora_r,
                        num_moe=self.num_lora_experts,
                        lora_alpha=self.lora_alpha,
                        lora_dropout=self.lora_dropout,
                        fan_in_fan_out=False,
                        bias=module.bias is not None,
                        gate_weights=self.gate_weights, # Pass the shared list
                    ).to(module.weight.device, dtype=module.weight.dtype) # Explicitly set dtype
                    setattr(parent_module, child_name, new_module)

    def forward(self, *args, **kwargs):
        """
        Overrides the forward method to pass arguments to the wrapped model.
        Gate weights are now handled externally by the caller.
        """
        # The `gate_weights` are now expected to be set on this model's `gate_weights`
        # attribute directly by the caller (e.g., iLoRAModel) before this
        # forward pass is called. This avoids passing it through kwargs and makes
        # the state management more explicit.
        if 'gate_weights' in kwargs:
            kwargs.pop('gate_weights')
            warnings.warn(
                "`gate_weights` was passed as a keyword argument to `MoeLoraModel.forward`, "
                "but it should be set directly on the `gate_weights` attribute. "
                "The argument has been ignored."
            )
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

class MoeLoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.num_moe = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, num_moe, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.num_moe[adapter_name] = num_moe
        self.lora_alpha[adapter_name] = lora_alpha
        
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            # iLoRA Reference Logic:
            # lora_A is (in_features, r) -> stored as Linear(in_features, r)
            # lora_B is (r, out_features) -> stored as Linear(r, out_features)
            # During forward, these are reshaped to split r into num_moe experts.
            
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            
            # Scaling: lora_alpha / (r // num_moe)
            # This matches reference: self.scaling[adapter_name] = lora_alpha / (r // num_moe)
            self.scaling[adapter_name] = lora_alpha / (r // num_moe)
            
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

class Linear(nn.Linear, MoeLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        num_moe: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        gate_weights: List = [],
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MoeLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        
        self.weight.requires_grad = False
        if kwargs.get("bias", False) and self.bias is not None:
             self.bias.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        
        self.update_layer(adapter_name, r, num_moe, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.gate_weights = gate_weights

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys() or self.disable_adapters or self.r[self.active_adapter] == 0:
            return F.linear(x, self.weight, bias=self.bias)
        
        if not self.merged and self.r[self.active_adapter] > 0:
            result = F.linear(x, self.weight, bias=self.bias)
            
            if not self.gate_weights:
                 raise ValueError("Gate weights are not set. Ensure gate_weights are passed to the model's forward method.")
            gate_weights = self.gate_weights[0]

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)
            batch_size, seq_len, _ = x.size()

            # Reference Forward Logic:
            # 1. Reshape weights to (num_experts, r_per_expert, features)
            # lora_A weight: (r, in_features) -> (num_experts, r_per_expert, in_features)
            # lora_B weight: (out_features, r) -> (num_experts, out_features, r_per_expert)
            
            r = self.r[self.active_adapter]
            num_moe = self.num_moe[self.active_adapter]
            r_per_expert = r // num_moe
            
            lora_A_weight = self.lora_A[self.active_adapter].weight.view(num_moe, r_per_expert, self.in_features)
            lora_B_weight = self.lora_B[self.active_adapter].weight.view(num_moe, self.out_features, r_per_expert)
            
            # 2. Compute LoRA output per expert
            # x: (batch, seq, in)
            # lora_A: (num, r_per, in)
            # einsum 'bse,nre->bsnr' -> (batch, seq, num, r_per)
            lora_output = torch.einsum('bse,nre->bsnr', x, lora_A_weight)
            
            # lora_B: (num, out, r_per)
            # einsum 'bsnr,nor->bsno' -> (batch, seq, num, out)
            lora_output = torch.einsum('bsnr,nor->bsno', lora_output, lora_B_weight)

            # 3. Apply gating
            # gate_weights: (batch, num)
            # einsum 'bsno,bn->bso' -> (batch, seq, out)
            # Note: gate_weights needs to be cast to match lora_output dtype
            gate_weights = gate_weights.to(lora_output.dtype)
            gated_output = torch.einsum('bsno,bn->bso', lora_output, gate_weights)
            
            result += gated_output * self.scaling[self.active_adapter]

        else:
            result = F.linear(x, self.weight, bias=self.bias)

        result = result.to(previous_dtype)
        return result