
import torch
import torch.nn as nn
from src.teacher.moe_lora_model import MoeLoraLayer, Linear
import time

def test_moelora_layer_memory():
    print("\n--- Testing MoeLoraLayer Memory Usage ---")
    
    # Configuration mimicking Qwen 1.8B
    batch_size = 16
    seq_len = 512
    in_features = 2048
    out_features = 2048
    num_experts = 4
    lora_r = 8
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create Layer
    layer = Linear(
        adapter_name="default",
        in_features=in_features,
        out_features=out_features,
        r=lora_r,
        num_moe=num_experts,
        lora_alpha=32,
        lora_dropout=0.1,
        gate_weights=[] # Will set manually
    ).to(device, dtype=torch.bfloat16)
    
    # Dummy Input
    x = torch.randn(batch_size, seq_len, in_features, device=device, dtype=torch.bfloat16)
    gate_weights = torch.randn(batch_size, num_experts, device=device, dtype=torch.bfloat16)
    layer.gate_weights = [gate_weights]
    
    # Measure Memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
    
    # Forward Pass
    output = layer(x)
    
    if device == "cuda":
        mem_after = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"Memory Before: {mem_before / 1024**2:.2f} MB")
        print(f"Memory After: {mem_after / 1024**2:.2f} MB")
        print(f"Peak Memory: {peak_mem / 1024**2:.2f} MB")
        print(f"Activation Overhead: {(peak_mem - mem_before) / 1024**2:.2f} MB")
    else:
        print("Running on CPU, cannot measure exact GPU memory usage.")
        # Theoretical calculation
        # x: B * S * H * 2 bytes
        input_size = batch_size * seq_len * in_features * 2
        # bsno tensor: B * S * N * Out * 2 bytes (if r_per_expert is small, bottleneck is here)
        # Wait, the code does:
        # lora_output = einsum('bse,nre->bsnr', x, lora_A) -> B * S * N * (r/N)
        # lora_output = einsum('bsnr,nor->bsno', lora_output, lora_B) -> B * S * N * Out
        
        intermediate_size = batch_size * seq_len * num_experts * out_features * 2
        print(f"Theoretical Input Size: {input_size / 1024**2:.2f} MB")
        print(f"Theoretical Intermediate Tensor (bsno) Size: {intermediate_size / 1024**2:.2f} MB")

if __name__ == "__main__":
    try:
        test_moelora_layer_memory()
    except Exception as e:
        print(f"Error: {e}")
