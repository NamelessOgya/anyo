import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from src.teacher.ilora_model import iLoRAModel
from src.teacher.mlp_projector import MLPProjector
from src.teacher.moe_lora_model import MoeLoraModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_checkpointing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Load Config & Model
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    print(f"Loading {model_name}...")
    
    # Load LLM
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        # attn_implementation="flash_attention_2"
    )
    
    # Enable Checkpointing MANUALLY to test
    print("Enabling Gradient Checkpointing...")
    llm.gradient_checkpointing_enable()
    print(f"LLM is_gradient_checkpointing flag: {getattr(llm, 'is_gradient_checkpointing', 'Not Found')}")
    print(f"LLM config use_cache: {llm.config.use_cache}") # Should be False usually if enabled? Or ignored?
    
    # Freeze params
    for param in llm.parameters():
        param.requires_grad = False
        
    # Wrap with MoeLora
    print("Wrapping with MoeLoraModel...")
    llm = MoeLoraModel(
        model=llm,
        target_modules=["q_proj", "v_proj"],
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        num_lora_experts=4,
    ).to(device)
    
    # Check flag again after wrapping
    # MoeLoraModel wraps llm in self.model
    print(f"Wrapped LLM is_gradient_checkpointing flag: {getattr(llm.model, 'is_gradient_checkpointing', 'Not Found')}")
    
    # Dummy components
    num_items = 100
    hidden_size = 64
    rec_model = nn.Linear(hidden_size, hidden_size).to(device) # Dummy
    rec_model.hidden_size = hidden_size # Add attribute
    # Mock rec_model methods
    rec_model.get_full_sequence_representations = lambda s, l: torch.randn(s.shape[0], s.shape[1], hidden_size, device=device, dtype=torch.bfloat16)
    rec_model._get_last_item_representation = lambda s, l: torch.randn(s.shape[0], hidden_size, device=device, dtype=torch.bfloat16)
    rec_model.item_embeddings = nn.Embedding(num_items+1, hidden_size).to(device)
    
    projector = MLPProjector(hidden_size, llm.model.config.hidden_size, hidden_size, 0.1).to(device, dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[HistoryEmb]', '[CansEmb]']})
    
    model = iLoRAModel(
        llm=llm.model, # Pass the INNER model? No, iLoRAModel expects the raw LLM and wraps it itself?
                       # WAIT. src/teacher/factory.py wraps it BEFORE passing to iLoRAModel?
                       # Let's check factory.py logic.
                       # factory.py:
                       #   llm = AutoModel...
                       #   llm.gradient_checkpointing_enable()
                       #   model = iLoRAModel(llm=llm, ...)
                       #
                       # src/teacher/ilora_model.py:
                       #   self.llm = MoeLoraModel(model=llm, ...)
                       #
                       # So factory passes the RAW LLM. iLoRAModel wraps it.
                       # My debug script above wrapped it manually. Let's match factory logic.
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
        candidate_topk=10,
        item_id_to_name={i: f"Item {i}" for i in range(num_items+1)},
        padding_item_id=0,
        llm_dtype=torch.bfloat16
    )
    
    # Re-create model using factory logic (pass raw LLM)
    # But wait, I already wrapped it above.
    # Let's restart the logic to match factory.py exactly.
    print("\n--- Re-simulating Factory Logic ---")
    llm_raw = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        # attn_implementation="flash_attention_2"
    )
    llm_raw.gradient_checkpointing_enable()
    
    # Freeze
    for param in llm_raw.parameters():
        param.requires_grad = False
        
    model = iLoRAModel(
        llm=llm_raw,
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
        candidate_topk=10,
        item_id_to_name={i: f"Item {i}" for i in range(num_items+1)},
        padding_item_id=0,
        llm_dtype=torch.bfloat16
    )
    model.to(device)
    model.train() # Set training mode
    
    print(f"Final Model LLM is_gradient_checkpointing: {getattr(model.llm.model, 'is_gradient_checkpointing', 'Not Found')}")
    print(f"Final Model LLM config gradient_checkpointing: {getattr(model.llm.model.config, 'gradient_checkpointing', 'Not Found')}")
    print(f"Final Model LLM training: {model.llm.model.training}")
    
    # 2. Run Forward/Backward
    batch_size = 4 # Small batch for debug
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
    
    print("\nRunning Forward...")
    # Hook to check inputs_embeds requires_grad
    def check_hook(module, inputs, output):
        print(f"LLM Input requires_grad: {inputs[0].requires_grad if inputs[0] is not None else 'None'}")
    
    handle = model.llm.model.register_forward_hook(check_hook)
    
    # torch.cuda.reset_peak_memory_stats()
    outputs = model.get_teacher_outputs(batch)
    loss = outputs["ranking_scores"].mean()
    
    mem_fwd = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Forward Peak Memory: {mem_fwd:.2f} GB")
    
    print("Running Backward...")
    loss.backward()
    
    mem_bwd = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Backward Peak Memory: {mem_bwd:.2f} GB")
    
    # Check if gradients exist on LoRA
    for name, param in model.llm.named_parameters():
        if "lora_A" in name and param.grad is not None:
            print(f"Gradient found on {name}")
            break
    else:
        print("WARNING: No gradients found on LoRA layers!")

if __name__ == "__main__":
    debug_checkpointing()
