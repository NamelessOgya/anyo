import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.dummies import DummySASRec, DummyGenerativeRanker

def test_dummy_sasrec():
    print("Testing DummySASRec...")
    num_items = 100
    hidden_size = 64
    max_seq_len = 20
    batch_size = 4
    teacher_embedding_dim = 128

    model = DummySASRec(num_items, hidden_size, max_seq_len, teacher_embedding_dim=teacher_embedding_dim)
    
    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len))
    item_seq_len = torch.randint(1, max_seq_len, (batch_size,))

    # Test forward
    output = model(item_seq, item_seq_len)
    print(f"Forward output shape: {output.shape}")
    assert output.shape == (batch_size, teacher_embedding_dim)

    # Test predict
    scores = model.predict(item_seq, item_seq_len)
    print(f"Predict scores shape: {scores.shape}")
    assert scores.shape == (batch_size, num_items + 1)

    # Test get_full_sequence_representations
    full_seq = model.get_full_sequence_representations(item_seq, item_seq_len)
    print(f"Full sequence representations shape: {full_seq.shape}")
    assert full_seq.shape == (batch_size, max_seq_len, hidden_size)
    print("DummySASRec passed!\n")

def test_dummy_generative_ranker():
    print("Testing DummyGenerativeRanker...")
    hidden_size = 4096
    model = DummyGenerativeRanker(hidden_size=hidden_size)
    
    prompts = ["Test prompt 1", "Test prompt 2"]
    
    # Test create_prompt
    prompt = model.create_prompt("history", ["cand1", "cand2"])
    print(f"Created prompt: {prompt}")
    assert isinstance(prompt, str)

    # Test generate_and_extract_state
    generated_texts, decision_states = model.generate_and_extract_state(prompts)
    print(f"Generated texts: {generated_texts}")
    print(f"Decision states shape: {decision_states.shape}")
    
    assert len(generated_texts) == len(prompts)
    assert decision_states.shape == (len(prompts), hidden_size)

    # Test parse_ranking
    ranking = model.parse_ranking(generated_texts[0], 2)
    print(f"Parsed ranking: {ranking}")
    assert isinstance(ranking, list)
    print("DummyGenerativeRanker passed!\n")

if __name__ == "__main__":
    test_dummy_sasrec()
    test_dummy_generative_ranker()
    print("All integration tests passed!")
