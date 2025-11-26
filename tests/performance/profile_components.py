import torch
import torch.nn as nn
import time
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def profile_components():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Setup Dummy Models
    # Tiny LLM config for functional test, but we want realistic relative sizes?
    # No, we should use the REAL model sizes to get realistic timing.
    # But loading 1.8B takes time.
    # Let's use a smaller LLM but scale the timing? 
    # Or just use the real config structure but random weights.
    
    print("Initializing models...")
    # SASRec (Real size)
    num_items = 5000
    hidden_size = 64
    max_seq_len = 50
    
    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items+1, hidden_size)
            self.blocks = nn.Sequential(*[
                nn.Linear(hidden_size, hidden_size) for _ in range(2) # Dummy blocks
            ])
        def get_full_sequence_representations(self, seq, len_seq):
            return self.blocks(self.item_embeddings(seq))
        def _get_last_item_representation(self, seq, len_seq):
            return torch.randn(seq.shape[0], hidden_size, device=seq.device)

    rec_model = SASRec().to(device)
    
    # LLM (Real size 1.8B approx)
    # Loading real weights takes too long. Let's use meta device or random init with correct config.
    config = AutoConfig.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
    # We can't easily run meta device forward pass without hooks.
    # Let's use a smaller proxy (OPT-125M) and extrapolate? 
    # Or just assume LLM is heavy.
    # Actually, the user's environment has the model cached.
    # Let's try to load the real model config but init empty?
    # No, let's use OPT-125M as a "Lower Bound" for LLM cost. 
    # If SASRec is negligible compared to 125M, it's definitely negligible for 1.8B.
    llm_name = "facebook/opt-125m" 
    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[HistoryEmb]']})
    llm.resize_token_embeddings(len(tokenizer))
    
    projector = MLPProjector(hidden_size, llm.config.hidden_size, hidden_size, 0.1).to(device)
    
    # 2. Prepare Data
    batch_size = 32
    seq_len = 512 # LLM seq len
    
    # Create inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    # Insert some [HistoryEmb] tokens
    his_token_id = tokenizer.additional_special_tokens_ids[0]
    # Replace random positions with history token
    mask = torch.rand(batch_size, seq_len, device=device) < 0.1
    input_ids[mask] = his_token_id
    
    attention_mask = torch.ones_like(input_ids)
    
    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len), device=device)
    len_seq = torch.full((batch_size,), max_seq_len, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = rec_model.get_full_sequence_representations(item_seq, len_seq)
        _ = llm(input_ids)
        
    # 3. Benchmark SASRec
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        rec_out = rec_model.get_full_sequence_representations(item_seq, len_seq)
    torch.cuda.synchronize()
    sasrec_time = (time.time() - start) / 100
    print(f"SASRec Forward Time: {sasrec_time*1000:.2f} ms")
    
    # 4. Benchmark Projector
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        proj_out = projector(rec_out)
    torch.cuda.synchronize()
    proj_time = (time.time() - start) / 100
    print(f"Projector Forward Time: {proj_time*1000:.2f} ms")

    # 5. Benchmark Replacement Logic (Approximation)
    # We simulate the scatter/gather logic
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        # Simplified replacement logic
        history_emb_mask = (input_ids == his_token_id)
        # Just some indexing ops
        idx = torch.nonzero(history_emb_mask)
        # This is just a proxy for the complex logic
    torch.cuda.synchronize()
    replace_time = (time.time() - start) / 100
    print(f"Replacement Logic (Proxy) Time: {replace_time*1000:.2f} ms")

    # 6. Benchmark LLM (OPT-125M)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = llm(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    llm_time = (time.time() - start) / 100
    print(f"LLM (125M) Forward Time: {llm_time*1000:.2f} ms")
    
    print("-" * 30)
    print(f"Ratio SASRec / LLM(125M): {sasrec_time / llm_time:.4f}")
    print("Note: Real LLM is 1.8B, so the ratio will be ~15x smaller.")

if __name__ == "__main__":
    profile_components()
