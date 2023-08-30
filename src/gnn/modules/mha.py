import torch
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange
from typing import Tuple, Optional

__all__ = ["MultiHeadAttention", "GraphMultiHeadAttention"]


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k if d_k is not None else d_model // n_head
        self.d_v = d_v if d_v is not None else d_model // n_head

        self.fc_qkv = nn.Linear(
            d_model,
            n_head * (2 * self.d_k + self.d_v),
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_proj = nn.Linear(
            n_head * self.d_v,
            d_model,
            bias=False,
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input (N T C)

        Returns:
            output (Tensor): output (N T C)
            weight (Tensor): attention weight (N n_head T T)
        """
        # compute query, key, value
        q, k, v = self.compute_qkv(x)

        # attention weight: softmax(qk/sqrt(d_k))
        qv = torch.matmul(q, k.transpose(2, 3)) / self.d_k**0.5
        weights = F.softmax(qv, dim=-1)  # N n_head T T

        # softmax(qk/sqrt(d_k))v
        x_attn = torch.matmul(weights, v)

        # concat and project each head
        x_attn = rearrange(x_attn, "N n_head T d_k -> N T (n_head d_k)")
        x_attn = self.fc_proj(x_attn)

        x_attn = self.dropout(x_attn)

        # residual connection & layer normalization
        output = self.layer_norm(x + x_attn)

        return output, weights

    def compute_qkv(self, x: Tensor) -> Tuple[Tensor]:
        """conpute Query Key Value
        Args:
            x (Tensor): input (N, T, C)

        Returns:
            q (Tensor): query (N n_head T d_k)
            k (Tensor): key   (N n_head T d_k)
            v (Tensor): vaue  (N n_head T d_v)
        """
        qkv = self.fc_qkv(x)

        q, k, v = torch.split(
            qkv,
            [self.n_head * self.d_k, self.n_head * self.d_k, self.n_head * self.d_v],
            dim=-1,
        )
        q = rearrange(q, "N T (n_head d_k) -> N n_head T d_k", n_head=self.n_head)
        k = rearrange(k, "N T (n_head d_k) -> N n_head T d_k", n_head=self.n_head)
        v = rearrange(v, "N T (n_head d_v) -> N n_head T d_v", n_head=self.n_head)
        return q, k, v


class GraphMultiHeadAttention(MultiHeadAttention):
    """Multi Head Attention class for Spatial Graph Convolutional Network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_head: int,
        dropout: float = 0.1,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
    ) -> None:
        super().__init__(
            d_model=in_channels, n_head=n_head, dropout=dropout, d_k=d_k, d_v=d_v
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc_proj = nn.Linear(
            n_head * self.d_v,
            out_channels,
            bias=False,
        )

        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input (N V C_in)
            A (Tesnor): adjacency matrix (1, V, V)

        Returns:
            output (Tensor): output (N T C_out)
            weight (Tensor): attention weight (N n_head T T)
        """
        # compute query, key, value
        q, k, v = self.compute_qkv(x)

        # attention weight: softmax(A + q*k)
        qv = torch.matmul(q, k.transpose(2, 3)) / self.d_k**0.5
        qv = rearrange(qv, "N n_head V1 V2 -> (N n_head) V1 V2")
        qv = qv + A.transpose(-2, -1)
        qv = rearrange(qv, "(N n_head) V1 V2 -> N n_head V1 V2", n_head=self.n_head)
        weights = F.softmax(qv, dim=-1)

        # softmax(qk/sqrt(d_k))v
        x_attn = torch.matmul(weights, v)

        # concat and project each head
        x_attn = rearrange(x_attn, "N n_head V d_k -> N V (n_head d_k)")
        x_attn = self.fc_proj(x_attn)
        x_attn = self.dropout(x_attn)

        # residual connection & layer normalization
        if self.in_channels == self.out_channels:
            output = self.layer_norm(x + x_attn)
        else:
            output = self.layer_norm(x_attn)
        return output, weights
