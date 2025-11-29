import torch

class BigRecCollator:
    def __init__(self, tokenizer, item_id_to_name, max_source_length=512, max_target_length=64, use_cot=False):
        self.tokenizer = tokenizer
        self.item_id_to_name = item_id_to_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_cot = use_cot

    def __call__(self, batch):
        # batch is a list of dicts from SASRecDataset
        # keys: item_seq, item_seq_len, next_item, (optional) reasoning
        
        instructions = []
        inputs = []
        outputs = []
        
        for item in batch:
            # 1. Format Input (History)
            seq = item["item_seq"]
            seq_len = item["item_seq_len"]
            # Filter padding (0) and get names
            hist_names = [self.item_id_to_name.get(idx.item(), f"Item_{idx.item()}") for idx in seq[:seq_len] if idx.item() > 0]
            input_text = ", ".join(hist_names)
            
            # 2. Format Output (Target)
            target_id = item["next_item"].item()
            target_text = self.item_id_to_name.get(target_id, f"Item_{target_id}")
            
            # Reasoning (Dummy or from batch)
            reasoning = item.get("reasoning", "")
            
            if self.use_cot:
                instructions.append("Recommend the next item for the user based on the history and explain the reasoning.")
                # If reasoning is provided, include it in the target output
                # Format: "Reasoning: {reasoning}\nRecommendation: {target}"
                # If no reasoning provided (e.g. during inference or if data missing), we might just expect model to generate it.
                # For training, we need ground truth reasoning.
                if reasoning:
                    outputs.append(f"Reasoning: {reasoning}\nRecommendation: {target_text}")
                else:
                    # Fallback or error? For now, let's assume we might want to train on just target if reasoning missing?
                    # Or maybe we use a dummy reasoning for testing pipeline.
                    outputs.append(f"Reasoning: [Reasoning]\nRecommendation: {target_text}")
            else:
                instructions.append("Recommend the next item for the user based on the history.")
                outputs.append(target_text)
            
            inputs.append(input_text)

        # 3. Tokenize
        prompts = []
        for inst, inp in zip(instructions, inputs):
            # BIGRec / Alpaca Prompt Format
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{inst}\n\n"
                f"### Input:\n{inp}\n\n"
                "### Response:\n"
            )
            prompts.append(prompt)
        
        # Re-doing tokenization for CausalLM standard
        full_sequences = [p + o + self.tokenizer.eos_token for p, o in zip(prompts, outputs)]
        
        tokenized_full = self.tokenizer(
            full_sequences,
            max_length=self.max_source_length + self.max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized_full.input_ids
        attention_mask = tokenized_full.attention_mask
        labels = input_ids.clone()
        
        # Mask out the prompt part in labels
        # We also want to return prompt_input_ids for inference
        tokenized_prompts = self.tokenizer(
            prompts,
            max_length=self.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        prompt_input_ids = tokenized_prompts.input_ids
        prompt_attention_mask = tokenized_prompts.attention_mask
        
        # We need lengths for masking labels in the full sequence
        # But tokenized_prompts is now padded, so we can't just use len(ids).
        # We need to use attention_mask sum.
        prompt_lengths = prompt_attention_mask.sum(dim=1)
        
        for i, length in enumerate(prompt_lengths):
            # Mask prompt in labels
            # Note: full_sequences might have different tokenization than prompt alone due to spacing/merging?
            # Usually okay if we use the same tokenizer.
            # But safer to use the logic: labels[:length] = -100
            labels[i, :length] = -100
            
        # Mask padding in full sequence
        labels[attention_mask == 0] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask
        }
