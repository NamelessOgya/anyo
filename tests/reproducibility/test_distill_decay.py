import pytest
import torch
from src.teacher.trainer_ilora import iLoRATrainer
from unittest.mock import MagicMock

def test_distill_decay_logic():
    # Mock Trainer to provide max_steps
    mock_trainer = MagicMock()
    mock_trainer.max_steps = 100
    
    # Common args
    lr = 1e-3
    wd = 0.0
    metrics_k = [10]
    
    # 1. Linear Decay
    # Start: 1.0, Min: 0.0
    trainer_linear = iLoRATrainer(
        ilora_model=MagicMock(),
        num_items=100,
        learning_rate=lr,
        weight_decay=wd,
        metrics_k=metrics_k,
        item_id_to_name={},
        distill_lambda=1.0,
        distill_min_lambda=0.0,
        distill_decay_type="linear"
    )
    trainer_linear.trainer = mock_trainer
    
    # Mock global_step using PropertyMock or by patching
    # LightningModule.global_step is a property that delegates to trainer.global_step
    # So we can just set trainer.global_step
    mock_trainer.global_step = 0
    
    # Step 0: 1.0
    mock_trainer.global_step = 0
    assert trainer_linear._get_current_lambda() == 1.0
    
    # Step 50: 0.5
    mock_trainer.global_step = 50
    assert trainer_linear._get_current_lambda() == 0.5
    
    # Step 100: 0.0
    mock_trainer.global_step = 100
    assert trainer_linear._get_current_lambda() == 0.0
    
    # 2. Cosine Decay
    # Start: 1.0, Min: 0.0
    trainer_cosine = iLoRATrainer(
        ilora_model=MagicMock(),
        num_items=100,
        learning_rate=lr,
        weight_decay=wd,
        metrics_k=metrics_k,
        item_id_to_name={},
        distill_lambda=1.0,
        distill_min_lambda=0.0,
        distill_decay_type="cosine"
    )
    trainer_cosine.trainer = mock_trainer
    
    # Step 0: 1.0
    mock_trainer.global_step = 0
    assert trainer_cosine._get_current_lambda() == 1.0
    
    # Step 50: 0.5 (Cosine(pi/2) = 0) -> 0 + 0.5 * 1 * (1 + 0) = 0.5
    mock_trainer.global_step = 50
    assert abs(trainer_cosine._get_current_lambda() - 0.5) < 1e-5
    
    # Step 100: 0.0 (Cosine(pi) = -1) -> 0 + 0.5 * 1 * (1 - 1) = 0.0
    mock_trainer.global_step = 100
    assert abs(trainer_cosine._get_current_lambda() - 0.0) < 1e-5
    
    # 3. None (No Decay)
    trainer_none = iLoRATrainer(
        ilora_model=MagicMock(),
        num_items=100,
        learning_rate=lr,
        weight_decay=wd,
        metrics_k=metrics_k,
        item_id_to_name={},
        distill_lambda=1.0,
        distill_min_lambda=0.0,
        distill_decay_type="none"
    )
    trainer_none.trainer = mock_trainer
    
    mock_trainer.global_step = 50
    assert trainer_none._get_current_lambda() == 1.0

    # 4. Fast Decay (distill_decay_steps < max_steps)
    # Start: 1.0, Min: 0.0, Decay Steps: 20, Max Steps: 100
    trainer_fast = iLoRATrainer(
        ilora_model=MagicMock(),
        num_items=100,
        learning_rate=lr,
        weight_decay=wd,
        metrics_k=metrics_k,
        item_id_to_name={},
        distill_lambda=1.0,
        distill_min_lambda=0.0,
        distill_decay_type="linear",
        distill_decay_steps=20
    )
    trainer_fast.trainer = mock_trainer
    
    # Step 0: 1.0
    mock_trainer.global_step = 0
    assert trainer_fast._get_current_lambda() == 1.0
    
    # Step 10: 0.5 (Halfway through 20 steps)
    mock_trainer.global_step = 10
    assert trainer_fast._get_current_lambda() == 0.5
    
    # Step 20: 0.0 (Reached end of decay)
    mock_trainer.global_step = 20
    assert trainer_fast._get_current_lambda() == 0.0
    
    # Step 50: 0.0 (Stay at min)
    mock_trainer.global_step = 50
    assert trainer_fast._get_current_lambda() == 0.0

    print("All decay logic tests passed!")
