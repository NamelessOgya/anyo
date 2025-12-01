import torch

class BigRecCollator:
    def __init__(self, tokenizer, item_id_to_name, max_source_length=512, max_target_length=64, use_cot=False, max_history_items=20, train_on_inputs=False, sasrec_max_seq_len=50):
        self.tokenizer = tokenizer
        self.item_id_to_name = item_id_to_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_cot = use_cot
        self.max_history_items = max_history_items
        self.train_on_inputs = train_on_inputs
        self.sasrec_max_seq_len = sasrec_max_seq_len

    def __call__(self, batch):
        # ... (lines 13-142 remain same)

        # Collect next_item_ids for evaluation
        next_item_ids = [item["next_item_id"] for item in batch]
        
        # Collect seq_ids for SASRec
        # SASRec expects right-padded sequences usually, but let's check SASRec implementation.
        # Assuming standard padding with 0.
        # Truncate to sasrec_max_seq_len (keep last N items)
        seq_ids_list = []
        for item in batch:
            seq = item["seq_ids"]
            if len(seq) > self.sasrec_max_seq_len:
                seq = seq[-self.sasrec_max_seq_len:]
            seq_ids_list.append(torch.tensor(seq, dtype=torch.long))
            
        # Pad right for SASRec (standard)
        sasrec_input_ids = torch.nn.utils.rnn.pad_sequence(seq_ids_list, batch_first=True, padding_value=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "next_item": torch.tensor(next_item_ids),
            "sasrec_input_ids": sasrec_input_ids
        }
