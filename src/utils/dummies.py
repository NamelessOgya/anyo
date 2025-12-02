import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

class DummySASRec(nn.Module):
    """
    Dummy SASRec model for testing purposes.
    Mimics the interface of src/student/models.py:SASRec
    """
    def __init__(self, num_items: int, hidden_size: int, max_seq_len: int, teacher_embedding_dim: int = None, *args, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.teacher_embedding_dim = teacher_embedding_dim
        
        # Minimal parameters to make it a valid module
        output_dim = self.teacher_embedding_dim if self.teacher_embedding_dim else self.hidden_size
        self.dummy_linear = nn.Linear(1, output_dim)
        self.dummy_predict = nn.Linear(1, num_items + 1)
        self.dummy_seq = nn.Linear(1, hidden_size)

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, teacher_embeddings: torch.Tensor = None):
        batch_size = item_seq.shape[0]
        # Create a dummy input that requires grad (derived from params)
        # We can just use the linear layer on a dummy input
        dummy_in = torch.ones(batch_size, 1, device=item_seq.device)
        return self.dummy_linear(dummy_in)

    def predict(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor):
        batch_size = item_seq.shape[0]
        dummy_in = torch.ones(batch_size, 1, device=item_seq.device)
        return self.dummy_predict(dummy_in)

    def get_full_sequence_representations(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, teacher_embeddings: torch.Tensor = None):
        batch_size, seq_len = item_seq.shape
        dummy_in = torch.ones(batch_size, seq_len, 1, device=item_seq.device)
        return self.dummy_seq(dummy_in)


class DummyGenerativeRanker(nn.Module):
    """
    Dummy GenerativeRanker for testing purposes.
    Mimics the interface of src/teacher/generative_ranker.py:GenerativeRanker
    """
    def __init__(self, hidden_size: int = 4096, *args, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.dummy_param = nn.Parameter(torch.empty(0))

    def create_prompt(self, history_text: str, candidates: List[str]) -> str:
        return f"Dummy prompt for {history_text}"

    def generate_and_extract_state(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 128
    ) -> Tuple[List[str], torch.Tensor]:
        batch_size = len(prompts)
        # Generate dummy rankings like "[1] > [0]"
        generated_texts = ["[1] > [0]" for _ in range(batch_size)]
        # Generate dummy decision states (batch_size, hidden_size)
        decision_states = torch.randn(batch_size, self.hidden_size)
        return generated_texts, decision_states

    def parse_ranking(self, generated_text: str, num_candidates: int) -> List[int]:
        # Simple dummy parser
        return list(range(num_candidates))

class DummyLLM(nn.Module):
    """
    Dummy LLM for testing purposes.
    Mimics AutoModelForCausalLM.
    """
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.config = config if config else type('Config', (), {'hidden_size': 128, '_name_or_path': 'dummy'})()
        self.config = config if config else type('Config', (), {'hidden_size': 128, '_name_or_path': 'dummy'})()
        
        import weakref
        class InnerDummyModel(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent_ref = weakref.ref(parent)
            def forward(self, *args, **kwargs):
                parent = self.parent_ref()
                if parent is None:
                    raise RuntimeError("Parent DummyLLM is gone")
                return parent.forward(*args, **kwargs)
        
        self.model = InnerDummyModel(self)
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.device = torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_size = self.config.hidden_size
        vocab_size = 1000 # Dummy vocab size

        # Dummy logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Dummy hidden states
        # Tuple of (batch, seq, hidden) for each layer + embedding?
        # Usually it's (hidden_states_layer_0, ..., hidden_states_layer_N)
        # We just return one for the last layer
        hidden_states = (torch.randn(batch_size, seq_len, hidden_size),)

        loss = None
        if labels is not None:
            loss = torch.tensor(0.1, requires_grad=True)

        return type('ModelOutput', (), {
            'logits': logits,
            'hidden_states': hidden_states,
            'loss': loss
        })()
    
    def resize_token_embeddings(self, new_num_tokens):
        pass

    def get_input_embeddings(self):
        return nn.Embedding(1000, self.config.hidden_size)
