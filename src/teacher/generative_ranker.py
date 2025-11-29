import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Any
import re

class GenerativeRanker(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        context_size: int = 4096,
        window_size: int = 20,
    ):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.context_size = context_size
        
        # Load Model (Teacher is usually frozen)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False
            
        self.tokenizer = tokenizer
        
        # Regex for parsing output
        self.output_extraction_regex = r"\[(\d+)\]"

    def create_prompt(self, history_text: str, candidates: List[str]) -> str:
        """
        Creates a RankLLM-style prompt.
        candidates: List of item descriptions or titles.
        """
        num = len(candidates)
        prefix = f"I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {history_text}.\n"
        
        body = ""
        for i, cand in enumerate(candidates):
            body += f"[{i+1}] {cand}\n"
            
        suffix = f"Search Query: {history_text}.\nRank the {num} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1]. Answer concisely."
        
        return prefix + body + suffix

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Standard forward pass (not used for generation-based distillation usually, but kept for compatibility)
        """
        return self.llm(input_ids=input_ids, attention_mask=attention_mask)

    def generate_and_extract_state(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 128
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generates ranking and extracts the 'Decision State' (hidden state at the first generated item ID).
        
        Returns:
            generated_texts: List of generated strings (rankings).
            decision_states: Tensor of shape (Batch, HiddenDim).
        """
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.context_size).to(self.device)
        
        # Generate with hidden states
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_sequences = outputs.sequences
        # Remove input tokens to get only generated tokens
        generated_tokens = generated_sequences[:, inputs.input_ids.shape[1]:]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Extract Decision State
        # We want the hidden state corresponding to the FIRST generated token that indicates an item ID.
        # For simplicity in this initial implementation, we take the hidden state of the *first generated token*.
        # In RankLLM prompt, the model immediately outputs "[ID]". 
        # The hidden state *input* to generate the first token is the last hidden state of the prompt.
        # The hidden state *after* generating the first token (e.g. "[") is outputs.hidden_states[0].
        
        # According to the research: "1位のアイテムトークン [B] を生成した瞬間の最終隠れ層ベクトル"
        # Ideally, we want the state *after* the model has decided "It's [1]".
        # If the output is "[1]", it's multiple tokens: "[", "1", "]".
        # The decision "It is 1" happens when generating "1".
        
        # Let's extract the hidden state of the first token generation step for now.
        # outputs.hidden_states is a tuple of tuples: (generated_steps, layers)
        # We want step 0 (first generated token), last layer.
        # shape: (Batch, 1, HiddenDim) if we take the last token of that step.
        
        # outputs.hidden_states[0] is the states for the first generated token.
        # It contains (Batch, SeqLen, Hidden) where SeqLen includes prompt? 
        # No, for generate, usually it returns states for the new token.
        # Let's assume outputs.hidden_states[0][-1] is (Batch, 1, Hidden).
        
        decision_states = outputs.hidden_states[0][-1][:, -1, :] # (Batch, Hidden)
        
        return generated_texts, decision_states

    def parse_ranking(self, generated_text: str, num_candidates: int) -> List[int]:
        """
        Parses "[2] > [1]" into [1, 0] (0-based indices).
        """
        matches = re.findall(self.output_extraction_regex, generated_text)
        ranking = []
        seen = set()
        for m in matches:
            idx = int(m) - 1 # 1-based to 0-based
            if 0 <= idx < num_candidates and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
        
        # Append missing items
        for i in range(num_candidates):
            if i not in seen:
                ranking.append(i)
                
        return ranking
