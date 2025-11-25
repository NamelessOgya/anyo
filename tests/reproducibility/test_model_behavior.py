import pytest
import torch
import torch.nn as nn
from src.student.models import SASRec
from src.teacher.ilora_model import iLoRAModel
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.teacher.mlp_projector import MLPProjector

@pytest.fixture(scope="module")
def model_behavior_fixture():
    """
    Fixture for Model Behavior tests.
    Sets up student and teacher models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_items = 100
    hidden_size = 32
    max_seq_len = 20
    
    student_model = SASRec(
        num_items=num_items,
        hidden_size=hidden_size,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1, # Enable dropout for determinism test
        max_seq_len=max_seq_len,
        padding_item_id=0
    ).to(device)
    
    # Minimal iLoRAModel setup
    llm_model_name = "facebook/opt-125m"
    llm = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))
    
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size, num_items):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_items = num_items
            
        def get_full_sequence_representations(self, item_seq, item_seq_len):
            batch_size, seq_len = item_seq.shape
            return torch.randn(batch_size, seq_len, self.hidden_size).to(item_seq.device)

        def _get_last_item_representation(self, item_seq, item_seq_len):
            batch_size = item_seq.shape[0]
            return torch.randn(batch_size, self.hidden_size).to(item_seq.device)

    dummy_rec_model = DummyRecModel(hidden_size, num_items).to(device)
    
    teacher_model = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=2,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        num_items=num_items,
        hidden_size=hidden_size,
        dropout_rate=0.0,
        rec_model=dummy_rec_model,
        projector=MLPProjector(hidden_size, llm.config.hidden_size, hidden_size, 0.0),
        candidate_topk=5,
        item_id_to_name={i: str(i) for i in range(num_items + 1)},
        padding_item_id=0,
        llm_dtype=torch.float32
    ).to(device)
    
    return {
        "student_model": student_model,
        "teacher_model": teacher_model,
        "device": device,
        "num_items": num_items,
        "max_seq_len": max_seq_len
    }

def test_21_batch_independence(model_behavior_fixture):
    """
    Test 21: [Batch Independence] Verify that output for a sample is independent of other samples in the batch.
    """
    model = model_behavior_fixture["student_model"]
    device = model_behavior_fixture["device"]
    model.eval() # Disable dropout
    
    # Sample 1
    seq1 = torch.randint(1, 100, (1, 20)).to(device)
    len1 = torch.tensor([10]).to(device)
    
    # Sample 2
    seq2 = torch.randint(1, 100, (1, 20)).to(device)
    len2 = torch.tensor([15]).to(device)
    
    # Run individually
    out1_single = model.predict(seq1, len1)
    out2_single = model.predict(seq2, len2)
    
    # Run batched
    seq_batch = torch.cat([seq1, seq2], dim=0)
    len_batch = torch.cat([len1, len2], dim=0)
    out_batch = model.predict(seq_batch, len_batch)
    
    # Check equality
    assert torch.allclose(out1_single, out_batch[0:1], atol=1e-5)
    assert torch.allclose(out2_single, out_batch[1:2], atol=1e-5)

def test_22_inference_determinism(model_behavior_fixture):
    """
    Test 22: [Inference Determinism] Verify that model.eval() produces deterministic outputs.
    """
    model = model_behavior_fixture["student_model"]
    device = model_behavior_fixture["device"]
    model.eval()
    
    seq = torch.randint(1, 100, (2, 20)).to(device)
    length = torch.tensor([10, 15]).to(device)
    
    out1 = model.predict(seq, length)
    out2 = model.predict(seq, length)
    
    assert torch.equal(out1, out2)

def test_23_edge_cases_short_sequence(model_behavior_fixture):
    """
    Test 23: [Edge Cases - Short Sequence] Verify handling of short sequences (length 1).
    """
    model = model_behavior_fixture["student_model"]
    device = model_behavior_fixture["device"]
    model.eval()
    
    # Length 1 sequence
    seq = torch.zeros((1, 20), dtype=torch.long).to(device)
    seq[0, 0] = 1 # First item is 1, rest padding
    length = torch.tensor([1]).to(device)
    
    # Should not error
    out = model.predict(seq, length)
    assert out.shape == (1, model.num_items) # SASRec.predict returns scores for valid items only
    # SASRec predict returns (batch_size, num_items) usually, but let's check fixture config
    # Fixture uses SASRec which returns (batch_size, num_items)
    # Wait, SASRec.predict returns scores against valid_item_embeds (num_items).
    # So shape is (1, 100).
    
    assert not torch.isnan(out).any()

def test_24_numerical_stability(model_behavior_fixture):
    """
    Test 24: [Numerical Stability] Verify no NaN/Inf for standard inputs.
    """
    model = model_behavior_fixture["student_model"]
    device = model_behavior_fixture["device"]
    model.train() # Enable dropout to check stability during training mode
    
    seq = torch.randint(1, 100, (4, 20)).to(device)
    length = torch.randint(1, 21, (4,)).to(device)
    
    out = model.predict(seq, length)
    
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_25_device_placement(model_behavior_fixture):
    """
    Test 25: [Device Placement] Verify all parameters are on the correct device.
    """
    model = model_behavior_fixture["student_model"]
    device = model_behavior_fixture["device"]
    
    for param in model.parameters():
        assert param.device == device
        
    # Check buffers too
    for buffer in model.buffers():
        assert buffer.device == device

def test_36_sasrec_output_dimension(model_behavior_fixture):
    """
    Test 36: [SASRec Output Dimension] Verify that SASRec.predict returns logits of shape (batch_size, num_items).
    It should NOT include padding index (0) in the output.
    """
    model = model_behavior_fixture["student_model"]
    device = model_behavior_fixture["device"]
    model.eval()
    
    batch_size = 2
    seq = torch.randint(1, model.num_items, (batch_size, 20)).to(device)
    length = torch.tensor([10, 15]).to(device)
    
    out = model.predict(seq, length)
    
    # Expected shape: (batch_size, num_items)
    assert out.shape == (batch_size, model.num_items)
    
    # Verify that we can compute CrossEntropyLoss with target in range [0, num_items-1]
    target = torch.randint(0, model.num_items, (batch_size,)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, target)
    assert not torch.isnan(loss)

    assert not torch.isnan(loss)

def test_38_ilora_output_behavior(model_behavior_fixture):
    """
    Test 38: [iLoRA Output Behavior] Verify that iLoRAModel.get_teacher_outputs returns correct keys and shapes.
    """
    model = model_behavior_fixture["teacher_model"]
    device = model_behavior_fixture["device"]
    num_items = model_behavior_fixture["num_items"]
    max_seq_len = model_behavior_fixture["max_seq_len"]
    
    batch_size = 2
    llm_seq_len = 10
    
    # Create dummy batch
    batch = {
        "seq": torch.randint(1, num_items, (batch_size, max_seq_len)).to(device),
        "len_seq": torch.tensor([5, 8]).to(device),
        "input_ids": torch.randint(0, 1000, (batch_size, llm_seq_len)).to(device),
        "attention_mask": torch.ones((batch_size, llm_seq_len)).to(device),
    }
    
    # Run get_teacher_outputs
    outputs = model.get_teacher_outputs(batch)
    
    # Verify keys
    expected_keys = {"ranking_scores", "embeddings", "candidates", "confidence"}
    assert set(outputs.keys()) == expected_keys
    
    # Verify shapes
    # ranking_scores: (batch_size, num_items)
    assert outputs["ranking_scores"].shape == (batch_size, num_items)
    
    # embeddings: (batch_size, llm_hidden_size)
    # llm_hidden_size depends on the model (OPT-125m is 768)
    assert outputs["embeddings"].shape == (batch_size, model.llm.model.config.hidden_size)
    
    # candidates: (batch_size, candidate_topk)
    assert outputs["candidates"].shape == (batch_size, model.candidate_topk)
    
    # confidence: (batch_size, candidate_topk)
    assert outputs["confidence"].shape == (batch_size, model.candidate_topk)
    
    # Verify values range
    assert (outputs["confidence"] >= 0).all() and (outputs["confidence"] <= 1).all()
    # Candidates should be valid item indices (0 to num_items-1)
    # Note: item_prediction_head maps to num_items. If items are 1-indexed, this might be 0..num_items-1 mapping to 1..num_items?
    # Usually it's 0-indexed logits corresponding to 1-indexed items if consistent with SASRec.predict logic.
    assert (outputs["candidates"] >= 0).all() and (outputs["candidates"] < num_items).all()

def test_39_ilora_training_step_logic(model_behavior_fixture):
    """
    Test 39: [iLoRA Training Step Logic] Verify that we can compute CrossEntropyLoss using ranking_scores.
    """
    model = model_behavior_fixture["teacher_model"]
    device = model_behavior_fixture["device"]
    num_items = model_behavior_fixture["num_items"]
    
    batch_size = 2
    llm_seq_len = 10
    
    # Create dummy batch
    batch = {
        "seq": torch.randint(1, num_items, (batch_size, 20)).to(device),
        "len_seq": torch.tensor([5, 8]).to(device),
        "input_ids": torch.randint(0, 1000, (batch_size, llm_seq_len)).to(device),
        "attention_mask": torch.ones((batch_size, llm_seq_len)).to(device),
        "next_item": torch.randint(1, num_items + 1, (batch_size,)).to(device) # 1-based IDs
    }
    
    # Run get_teacher_outputs
    outputs = model.get_teacher_outputs(batch)
    ranking_scores = outputs["ranking_scores"] # (batch_size, num_items)
    
    # Target items (1-based)
    target_items = batch["next_item"]
    
    # Check shape mismatch if any
    assert ranking_scores.shape[0] == target_items.shape[0]
    
    # Calculate CrossEntropyLoss
    # ranking_scores corresponds to items 1..N (indices 0..N-1)
    # target_items are 1..N
    # So target for loss should be target_items - 1
    loss = nn.functional.cross_entropy(ranking_scores, target_items - 1)
    
    assert not torch.isnan(loss)
    assert loss.item() > 0
    
    # Verify gradients
    # Ensure model is in train mode
    model.train()
    # Zero grads
    model.zero_grad()
    
    # Re-run forward pass to ensure graph is built
    outputs = model.get_teacher_outputs(batch)
    ranking_scores = outputs["ranking_scores"]
    loss = nn.functional.cross_entropy(ranking_scores, target_items - 1)
    
    loss.backward()
    
    # Check if some parameters have gradients
    # item_prediction_head should definitely have gradients
    assert model.item_prediction_head.weight.grad is not None
    assert model.item_prediction_head.weight.grad.abs().sum() > 0
