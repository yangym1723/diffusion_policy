import torch
import torch.nn as nn


class HistoryActionEncoder(nn.Module):
    """Project cumulative action history into feature space."""
    def __init__(self, action_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        action_history: (B, T, action_dim)
        return: (B, T, output_dim)
        """
        return self.net(action_history)
