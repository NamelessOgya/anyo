import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
