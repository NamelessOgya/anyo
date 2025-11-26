
import pytest
import torch
import torch.nn as nn
from src.teacher.trainer_ilora import iLoRATrainer
from unittest.mock import MagicMock

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Frozen layer (simulating LLM base)
        self.frozen = nn.Linear(10, 10)
        for p in self.frozen.parameters():
            p.requires_grad = False
            
        # Trainable layer (simulating LoRA/Projector)
        self.trainable = nn.Linear(10, 10)
        
        # Buffer (simulating BatchNorm running stats - though LLM usually doesn't have it, good to test)
        self.register_buffer('dummy_buffer', torch.zeros(10))

def test_checkpoint_optimization():
    # Setup Trainer with Mock Model
    model = MockModel()
    
    # We need to mock iLoRAModel interface because iLoRATrainer expects it
    # But for this test, we just want to test on_save_checkpoint logic if we inject it.
    # Since we are modifying iLoRATrainer, we can test the logic directly if we implement it as a mixin or just patch it.
    
    # However, we want to test the actual implementation in iLoRATrainer.
    # So we should instantiate iLoRATrainer.
    # iLoRATrainer expects ilora_model.
    
    ilora_model = MagicMock()
    # We need to make sure ilora_model is an nn.Module so iLoRATrainer can register it
    # But iLoRATrainer assigns self.model = ilora_model.
    # If we pass our MockModel as ilora_model, it should work for parameter inspection.
    
    # But iLoRATrainer calls self.save_hyperparameters(ignore=["ilora_model"])
    # and expects specific args.
    
    trainer = iLoRATrainer(
        ilora_model=model, # Pass our MockModel instead of real iLoRAModel
        num_items=100,
        learning_rate=1e-3,
        weight_decay=0.0,
        metrics_k=10,
        item_id_to_name={}
    )
    
    # Create a dummy checkpoint state_dict
    # LightningModule state_dict keys will be prefixed with "model." because self.model = ilora_model
    # But wait, iLoRATrainer assigns self.model = ilora_model.
    # So parameters are accessible via trainer.model.frozen...
    # The state_dict keys will be "model.frozen.weight", "model.trainable.weight", etc.
    
    state_dict = trainer.state_dict()
    checkpoint = {"state_dict": state_dict}
    
    # Verify initial state
    assert "model.frozen.weight" in checkpoint["state_dict"]
    assert "model.trainable.weight" in checkpoint["state_dict"]
    assert "model.dummy_buffer" in checkpoint["state_dict"]
    
    # Run on_save_checkpoint (which we will implement)
    # Since it's not implemented yet, this test should FAIL to filter if we run it now,
    # or pass if we implement the logic in the test to verify it works, then move to code.
    
    # Let's define the logic we WANT to implement and test it here first.
    def on_save_checkpoint_implementation(trainer_instance, checkpoint):
        state_dict = checkpoint["state_dict"]
        
        # Identify trainable parameters
        trainable_param_names = {n for n, p in trainer_instance.named_parameters() if p.requires_grad}
        all_param_names = {n for n, p in trainer_instance.named_parameters()}
        
        keys_to_keep = []
        for key in state_dict.keys():
            if key in trainable_param_names:
                keys_to_keep.append(key)
            elif key not in all_param_names:
                # Keep buffers
                keys_to_keep.append(key)
        
        new_state_dict = {k: v for k, v in state_dict.items() if k in keys_to_keep}
        checkpoint["state_dict"] = new_state_dict

    # Execute logic
    on_save_checkpoint_implementation(trainer, checkpoint)
    
    # Verify results
    assert "model.frozen.weight" not in checkpoint["state_dict"]
    assert "model.trainable.weight" in checkpoint["state_dict"]
    assert "model.dummy_buffer" in checkpoint["state_dict"]
    
    print("Checkpoint optimization logic verified: Frozen weights removed.")
    
    # Test Loading
    # Create a new model instance
    new_model = MockModel()
    new_trainer = iLoRATrainer(
        ilora_model=new_model,
        num_items=100,
        learning_rate=1e-3,
        weight_decay=0.0,
        metrics_k=10,
        item_id_to_name={}
    )
    
    # Load state_dict
    # Since we removed frozen weights, strict loading might fail if we don't handle it.
    # But PyTorch Lightning's load_from_checkpoint usually handles strict=False if specified.
    # Here we simulate loading manually.
    
    # When loading, we expect missing keys for frozen parameters.
    # But trainable parameters should load correctly.
    
    # Modify trainable weight in checkpoint to verify it actually loads
    checkpoint["state_dict"]["model.trainable.weight"].fill_(9.99)
    
    # Load
    # strict=False is required because frozen weights are missing in checkpoint but present in model
    missing_keys, unexpected_keys = new_trainer.load_state_dict(checkpoint["state_dict"], strict=False)
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # Verify missing keys are indeed the frozen ones
    assert "model.frozen.weight" in missing_keys
    assert "model.frozen.bias" in missing_keys
    
    # Verify trainable weights are loaded
    assert torch.allclose(new_trainer.model.trainable.weight, torch.tensor(9.99))
    
    # Verify unexpected keys is empty (we shouldn't have extra keys)
    assert len(unexpected_keys) == 0
    
    print("Checkpoint loading verified: Trainable weights loaded, frozen weights skipped.")

if __name__ == "__main__":
    test_checkpoint_optimization()
