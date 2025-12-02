import sys
import os
from omegaconf import OmegaConf
from src.core.config_utils import load_hydra_config

def main():
    print("DEBUG: Starting config debug script")
    overrides = sys.argv[1:]
    print(f"DEBUG: Overrides: {overrides}")
    
    print(f"DEBUG: CWD: {os.getcwd()}")
    try:
        cfg = load_hydra_config(config_path="../../conf", overrides=overrides)
        print("DEBUG: Config loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return

    # Check teacher config
    if "teacher" in cfg:
        print(f"DEBUG: cfg.teacher keys: {list(cfg.teacher.keys())}")
        
        val = cfg.teacher.get("compute_item_embeddings", "NOT_SET")
        print(f"DEBUG: cfg.teacher.compute_item_embeddings = {val}")
        
        model_type = cfg.teacher.get("model_type", "NOT_SET")
        print(f"DEBUG: cfg.teacher.model_type = {model_type}")
        
        item_emb_path = cfg.teacher.get("item_embeddings_path", "NOT_SET")
        print(f"DEBUG: cfg.teacher.item_embeddings_path = {item_emb_path}")
    else:
        print("ERROR: 'teacher' key not found in config")

if __name__ == "__main__":
    main()
