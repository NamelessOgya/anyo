
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.teacher.factory import create_teacher_model
from src.student.datamodule import SASRecDataModule
from src.teacher.trainer_ilora import iLoRATrainer

def check_trainable_parameters():
    """
    Initializes the teacher model and trainer to check which parameters are trainable.
    """
    # --- 1. Load Configurations ---
    print("Loading configurations...")
    # Load the main config.yaml
    # This will load the defaults structure, but not the actual content of the defaults
    cfg = OmegaConf.create() # Start with an empty config
    
    # Manually load and merge the default configs
    # This mimics Hydra's behavior for the relevant parts
    cfg.dataset = OmegaConf.load("conf/dataset/movielens.yaml")
    cfg.teacher = OmegaConf.load("conf/teacher/ilora.yaml")
    cfg.student = OmegaConf.load("conf/student/sasrec.yaml") # Load student config for max_seq_len
    cfg.train = OmegaConf.load("conf/train/teacher.yaml") # Assuming we are checking teacher training
    # Other configs (distill, eval) are not strictly needed for this check,
    # but can be loaded if desired for completeness or future use.

    llm_model_name = cfg.teacher.llm_model_name
    print(f"Using LLM: {llm_model_name}")

    # --- 2. Setup Tokenizer and LLM ---
    print("Setting up tokenizer and LLM...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = ['[PH]', '[HistoryEmb]', '[CansEmb]', '[ItemEmb]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
    llm.resize_token_embeddings(len(tokenizer))

    # --- 3. Setup DataModule ---
    print("Setting up DataModule...")
    # Instantiate SASRecDataModule to get num_items and item_id_to_name
    datamodule = SASRecDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size, # Use a dummy batch size for setup
        max_seq_len=cfg.student.max_seq_len,
        num_workers=0, # No need for workers during setup
        limit_data_rows=cfg.dataset.limit_data_rows,
        train_file="train.csv",
        val_file="val.csv",
        test_file="test.csv"
    )
    datamodule.setup() # This will populate num_items and item_id_to_name

    num_items_to_use = datamodule.num_items
    item_id_to_name_to_use = datamodule.item_id_to_name
    padding_item_id_to_use = datamodule.padding_item_id

    print(f"Actual num_items from DataModule: {num_items_to_use}")
    print(f"Padding item ID from DataModule: {padding_item_id_to_use}")

    # --- 4. Create Teacher Model using Factory ---
    print("Creating teacher model...")
    
    # The factory requires a valid checkpoint path. Let's use the one from the config.
    # We assume the student baseline has been run and a checkpoint exists.
    if not cfg.teacher.rec_model_checkpoint_path:
        raise ValueError("cfg.teacher.rec_model_checkpoint_path must be set in conf/teacher/ilora.yaml")

    teacher_model_instance = create_teacher_model(
        cfg=cfg,
        llm=llm,
        tokenizer=tokenizer,
        num_items=num_items_to_use,
        max_seq_len=cfg.student.max_seq_len,
        item_id_to_name=item_id_to_name_to_use,
        padding_item_id=padding_item_id_to_use,
        candidate_topk=10 # dummy value
    )

    # --- 5. Instantiate Trainer ---
    print("Instantiating trainer...")
    trainer = iLoRATrainer(
        ilora_model=teacher_model_instance,
        num_items=num_items_to_use,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        metrics_k=10 # dummy value
    )

    # --- 6. Check Trainable Parameters ---
    print("\n" + "="*50)
    print("Trainable Parameters:")
    print("="*50)
    total_trainable = 0
    total_params = 0
    for name, param in trainer.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(name)
            total_trainable += param.numel()
    
    print("\n" + "="*50)
    print(f"Total parameters: {total_params / 1_000_000:.2f}M")
    print(f"Trainable parameters: {total_trainable / 1_000_000:.2f}M")
    print(f"Trainable ratio: {100 * total_trainable / total_params:.4f}%")
    print("="*50)


if __name__ == "__main__":
    check_trainable_parameters()
