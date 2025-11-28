import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
import time
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def profile_memory(batch_size, use_checkpointing, use_compile):
    print(f"\n--- Profiling Batch Size: {batch_size}, Checkpointing: {use_checkpointing}, Compile: {use_compile} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available. Skipping.")
        return

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Start Memory: {start_mem:.2f} GB")

    # 1. Load Model (Dummy-ish)
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    llm.resize_token_embeddings(len(tokenizer))
    
    if use_checkpointing:
        llm.gradient_checkpointing_enable()

    # Dummy Rec Model & Projector
    hidden_size = 64
    num_items = 3706
    
    class DummyRecModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items + 1, hidden_size)
            self.hidden_size = hidden_size
        def get_full_sequence_representations(self, seq, len_seq):
            return torch.randn(seq.shape[0], seq.shape[1], hidden_size, device=seq.device)
        def _get_last_item_representation(self, seq, len_seq):
            return torch.randn(seq.shape[0], hidden_size, device=seq.device)

    rec_model = DummyRecModel()
    projector = MLPProjector(hidden_size, llm.config.hidden_size, hidden_size, 0.1)

    model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_items=num_items,
        num_lora_experts=4,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        hidden_size=hidden_size,
        dropout_rate=0.1,
        rec_model=rec_model,
        projector=projector,
        candidate_topk=20,
        item_id_to_name={},
        padding_item_id=0,
        llm_dtype=torch.bfloat16
    )
    
    if use_compile:
        model = torch.compile(model)

    model.to(device)
    
    load_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Model Loaded Memory: {load_mem:.2f} GB")

    # 2. Create Dummy Batch
    seq_len = 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    seq = torch.randint(1, num_items, (batch_size, 50), device=device)
    len_seq = torch.full((batch_size,), 50, device=device)
    next_item = torch.randint(1, num_items, (batch_size,), device=device)
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq": seq,
        "len_seq": len_seq,
        "next_item": next_item
    }

    # 3. Forward Pass
    print("Running Forward...")
    try:
        outputs = model.get_teacher_outputs(batch)
        ranking_scores = outputs["ranking_scores"]
        loss = F.cross_entropy(ranking_scores, next_item - 1)
        
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Forward Peak Memory: {fwd_mem:.2f} GB")
        
        # 4. Backward Pass
        print("Running Backward...")
        loss.backward()
        
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Backward Peak Memory: {bwd_mem:.2f} GB")
        
    except RuntimeError as e:
        print(f"OOM or Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    
    profile_memory(args.batch_size, args.checkpointing, args.compile)
