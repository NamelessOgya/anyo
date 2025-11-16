import torch
import torch.nn as nn
import torch.nn.functional as F
from src.third_party.dllm2rec.SASRecModules_ori import MultiHeadAttention, PositionwiseFeedForward
import logging
from typing import List

log = logging.getLogger(__name__)

# Helper function from DLLM2Rec/utility.py
def extract_axis_1(data, ind):
    batch_range = torch.arange(data.shape[0], device=data.device)
    return data[batch_range, ind]

class GRU(nn.Module):
    def __init__(self, item_num: int, hidden_size: int, state_size: int, dropout: float, device: torch.device, gru_layers: int = 1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.device = device

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1, # +1 for padding item
            embedding_dim=self.hidden_size,
            padding_idx=item_num # Assuming item_num is the padding index
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

        # This linear layer is for LLM embedding distillation, if llm_emb is 4096
        # It should be configurable or dynamically sized based on LLM emb size
        self.fc_llm_to_hidden = nn.Linear(4096, hidden_size) 
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, len_states: torch.Tensor, llm_emb: torch.Tensor = None, ed_weight: float = 0.0):
        emb = self.item_embeddings(states)
        
        if llm_emb is not None and ed_weight > 0:
            # Project LLM embeddings to student model's hidden size
            projected_llm_emb = self.fc_llm_to_hidden(llm_emb)
            emb = emb + ed_weight * projected_llm_emb
        
        emb = self.dropout_layer(emb)

        # Ensure len_states are not zero for pack_padded_sequence
        len_states_clamped = torch.clamp(len_states, min=1).cpu() # pack_padded_sequence expects CPU tensor for lengths
        
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states_clamped, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2]) # Take the last layer's hidden state
        supervised_output = self.s_fc(hidden)
        return supervised_output

class Caser(nn.Module):
    def __init__(self, item_num: int, hidden_size: int, state_size: int, dropout: float, device: torch.device, num_filters: int, filter_sizes: List[int]):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout
        self.device = device

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
            padding_idx=item_num
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc_llm_to_hidden = nn.Linear(4096, hidden_size)

    def forward(self, states: torch.Tensor, len_states: torch.Tensor, llm_emb: torch.Tensor = None, ed_weight: float = 0.0):
        input_emb = self.item_embeddings(states)
        
        if llm_emb is not None and ed_weight > 0:
            projected_llm_emb = self.fc_llm_to_hidden(llm_emb)
            input_emb = input_emb + ed_weight * projected_llm_emb
        
        # Mask padding items
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        
        input_emb = input_emb.unsqueeze(1) # Add channel dimension for Conv2d

        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = F.relu(cnn(input_emb))
            h_out = h_out.squeeze(-1) # Remove last dimension (hidden_size)
            p_out = F.max_pool1d(h_out, h_out.shape[2]).squeeze(-1) # Max pool over sequence length
            pooled_outputs.append(p_out)

        h_pool_flat = torch.cat(pooled_outputs, 1)

        v_out = F.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size) # Reshape to (batch_size, hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

class SASRec(nn.Module):
    def __init__(self, item_num: int, hidden_size: int, state_size: int, dropout: float, device: torch.device, num_heads: int = 1):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.dropout_rate = dropout
        self.device = device

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1, # +1 for padding item
            embedding_dim=hidden_size,
            padding_idx=item_num # Assuming item_num is the padding index
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size, # Max sequence length
            embedding_dim=hidden_size
        )
        
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        
        self.s_fc = nn.Linear(hidden_size, item_num)
           
        self.fc_llm_to_hidden = nn.Linear(4096, hidden_size) # For LLM embedding distillation

    def forward(self, states: torch.Tensor, len_states: torch.Tensor, llm_emb: torch.Tensor = None, ed_weight: float = 0.0):
        inputs_emb = self.item_embeddings(states)
        
        if llm_emb is not None and ed_weight > 0:
            projected_llm_emb = self.fc_llm_to_hidden(llm_emb)
            inputs_emb = inputs_emb + ed_weight * projected_llm_emb
        
        # Add positional embeddings
        positions = torch.arange(self.state_size, device=self.device).unsqueeze(0)
        inputs_emb += self.positional_embeddings(positions)
        
        seq = self.emb_dropout(inputs_emb)
        
        # Mask padding items
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)        
        seq *= mask
        
        seq_normalized = self.ln_1(seq)

        mh_attn_out = self.mh_attn(seq_normalized, seq)

        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask # Apply mask again after feed forward
        ff_out = self.ln_3(ff_out)  
        
        # Extract the last valid hidden state
        state_hidden = extract_axis_1(ff_out, len_states - 1) 
        supervised_output = self.s_fc(state_hidden).squeeze(1) # Squeeze if output is (batch_size, 1, item_num)

        return supervised_output

def get_student_model(model_name: str, item_num: int, hidden_size: int, state_size: int, dropout: float, device: torch.device, **kwargs) -> nn.Module:
    """
    Factory function to get a student model instance.
    """
    if model_name == "GRU":
        return GRU(item_num, hidden_size, state_size, dropout, device, **kwargs)
    elif model_name == "SASRec":
        return SASRec(item_num, hidden_size, state_size, dropout, device, **kwargs)
    elif model_name == "Caser":
        filter_sizes = kwargs.get('filter_sizes', [2,3,4])
        num_filters = kwargs.get('num_filters', 16)
        return Caser(item_num, hidden_size, state_size, dropout, device, num_filters, filter_sizes)
    else:
        raise ValueError(f"Unknown student model: {model_name}")

if __name__ == "__main__":
    # Example usage and testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item_num = 100
    hidden_size = 64
    state_size = 50 # max_seq_len
    dropout = 0.1
    batch_size = 4

    # Dummy input data
    # states: batch of sequences, padded with item_num (which is padding_idx)
    states = torch.randint(0, item_num, (batch_size, state_size), device=device)
    len_states = torch.randint(1, state_size + 1, (batch_size,), device=device)
    # Sort states by length in descending order for GRU pack_padded_sequence
    len_states, sorted_indices = len_states.sort(descending=True)
    states = states[sorted_indices]

    # Dummy LLM embeddings (batch_size, state_size, 4096)
    llm_emb = torch.randn(batch_size, state_size, 4096, device=device)

    print("--- Testing GRU Model ---")
    gru_model = get_student_model("GRU", item_num, hidden_size, state_size, dropout, device).to(device)
    gru_output = gru_model(states, len_states, llm_emb, ed_weight=0.5)
    print(f"GRU output shape: {gru_output.shape}")
    assert gru_output.shape == (batch_size, item_num)

    print("\n--- Testing Caser Model ---")
    caser_model = get_student_model("Caser", item_num, hidden_size, state_size, dropout, device, num_filters=16, filter_sizes=[2,3,4]).to(device)
    caser_output = caser_model(states, len_states, llm_emb, ed_weight=0.5)
    print(f"Caser output shape: {caser_output.shape}")
    assert caser_output.shape == (batch_size, item_num)

    print("\n--- Testing SASRec Model ---")
    sasrec_model = get_student_model("SASRec", item_num, hidden_size, state_size, dropout, device, num_heads=1).to(device)
    sasrec_output = sasrec_model(states, len_states, llm_emb, ed_weight=0.5)
    print(f"SASRec output shape: {sasrec_output.shape}")
    assert sasrec_output.shape == (batch_size, item_num)

    print("\nAll student models tested successfully!")
