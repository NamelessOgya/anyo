import torch

class BigRecCollator:
    def __init__(self, tokenizer, item_id_to_name, max_source_length=512, max_target_length=64, use_cot=False, max_history_items=20, train_on_inputs=False):
        self.tokenizer = tokenizer
        self.item_id_to_name = item_id_to_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_cot = use_cot
        self.max_history_items = max_history_items
        self.train_on_inputs = train_on_inputs

    def __call__(self, batch):
        # batch is a list of dicts from SASRecDataset
        # keys: seq_ids, next_item_id, ...
        
        instructions = []
        inputs = []
        outputs = []
        
        for item in batch:
            # 1. Format Input (History)
            seq = item["seq_ids"]
            # Truncate history to keep prompt length manageable
            if len(seq) > self.max_history_items:
                seq = seq[-self.max_history_items:]
            
            seq_len = len(seq)
            # Filter padding (0) and get names
            # Note: Reference wraps titles in quotes
            hist_names = [f'"{self.item_id_to_name.get(idx, f"Item_{idx}")}"' for idx in seq if idx > 0]
            
            # Reference Input Format: "The user has watched the following movies before: "Title1", "Title2", ..."
            # process.ipynb adds "\n " to the end of the input
            history_str = ", ".join(hist_names)
            input_text = f"The user has watched the following movies before: {history_str}\n "
            
            # 2. Format Output (Target)
            target_id = item["next_item_id"]
            # Reference Target Format: "Title" (quoted)
            target_name = self.item_id_to_name.get(target_id, f"Item_{target_id}")
            target_text = f'"{target_name}"'
            
            # Reasoning (Dummy or from batch)
            reasoning = item.get("reasoning", "")
            
            if self.use_cot:
                # Update instruction for CoT if needed, but for now matching reference non-CoT
                instructions.append("Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user and explain the reasoning.")
                if reasoning:
                    outputs.append(f"Reasoning: {reasoning}\nRecommendation: {target_text}")
                else:
                    outputs.append(f"Reasoning: [Reasoning]\nRecommendation: {target_text}")
            else:
                # Reference Instruction
                instructions.append("Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user.")
                outputs.append(target_text)
            
            inputs.append(input_text)

        # 3. Tokenize Parts Separately
        # This avoids issues where token merging across Prompt-Target boundary changes lengths
        
        # Tokenize Prompts (Left Padding for batch processing if needed, but we do it manually or let tokenizer handle it)
        # We want: [PAD, ..., PAD, Prompt, Target, EOS]
        # And Labels: [-100, ..., -100, -100, Target, EOS]
        
        # We process sample by sample to ensure correctness, then pad
        
        input_ids_list = []
        labels_list = []
        prompt_input_ids_list = [] # For inference
        
        for inst, inp, out in zip(instructions, inputs, outputs):
            # Prompt
            prompt_text = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. \n\n"
                f"### Instruction:\n{inst}\n\n"
                f"### Input:\n{inp}\n\n"
                "### Response:\n"
            )
            
            # Target
            target_text = out + self.tokenizer.eos_token
            
            # Tokenize
            # Note: add_special_tokens=True adds BOS if defined
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True, truncation=True, max_length=self.max_source_length)["input_ids"]
            target_ids = self.tokenizer(target_text, add_special_tokens=False, truncation=True, max_length=self.max_target_length)["input_ids"]
            
            # Concatenate
            full_ids = prompt_ids + target_ids
            
            # Create Labels
            if self.train_on_inputs:
                # If training on inputs, labels are same as input_ids
                label_ids = full_ids
            else:
                # Mask prompt
                label_ids = [-100] * len(prompt_ids) + target_ids
            
            # Truncate if total length exceeds limit (though we truncated parts already)
            # But prompt+target might exceed model max length?
            # We assume max_source + max_target is within model limits.
            
            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(label_ids, dtype=torch.long))
            prompt_input_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))
            
        # Pad
        # We use torch.nn.utils.rnn.pad_sequence
        # But we need left padding for input_ids (for generation)?
        # Actually, for training CausalLM, right padding is standard/fine if we mask pads.
        # But for inference (generation), left padding is required.
        # BigRecModel sets padding_side="left".
        # So we should pad left.
        
        # Helper for left padding
        def pad_left(tensors, padding_value):
            max_len = max(len(t) for t in tensors)
            padded = []
            for t in tensors:
                pad_len = max_len - len(t)
                pad = torch.full((pad_len,), padding_value, dtype=t.dtype)
                padded.append(torch.cat([pad, t]))
            return torch.stack(padded)

        input_ids = pad_left(input_ids_list, self.tokenizer.pad_token_id)
        labels = pad_left(labels_list, -100)
        
        # Attention Mask
        # 1 for real tokens, 0 for pads
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # For prompt_input_ids (used in validation generation), we also need left padding
        prompt_input_ids = pad_left(prompt_input_ids_list, self.tokenizer.pad_token_id)
        prompt_attention_mask = (prompt_input_ids != self.tokenizer.pad_token_id).long()
            
        # Collect next_item_ids for evaluation
        next_item_ids = [item["next_item_id"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "next_item": torch.tensor(next_item_ids)
        }
