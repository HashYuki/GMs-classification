from torch import Tensor
from torch import nn

__all__ = ["PositionwiseFeedForward"]


class PositionwiseFeedForward(nn.Module):
    """A fully connected feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input (N T C)

        Returns:
            output (Tensor): output (N, T, C)
        """
        res = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        output = self.layer_norm(res + x)

        return output
