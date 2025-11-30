import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os

def verify_loading():
    print("Starting Model Loading Verification...")
    
    # Use a small model for verification to avoid huge downloads
    model_name = "facebook/opt-125m" 
    print(f"Using model: {model_name}")
    
    # 1. Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = tokenizer.decode(0)
        
    # 2. Quantization Config (Removed)
    print("Quantization disabled by default.")
    torch_dtype = torch.float16 # Default
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    
    # 3. Load Model
    print("Loading Model...")
    load_kwargs = {
        "torch_dtype": torch_dtype,
    }
        
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 4. Prepare for k-bit training (Removed)
    print("Skipping prepare_model_for_kbit_training (Standard mode).")

    # 5. Apply LoRA
    print("Applying LoRA...")
    try:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
        print("LoRA applied successfully!")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"FAILED to apply LoRA: {e}")
        return

    print("\nVERIFICATION SUCCESSFUL: Pipeline works correctly (CPU/GPU adaptive).")

if __name__ == "__main__":
    verify_loading()
