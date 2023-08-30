import torch
from torch import Tensor, nn
from einops import rearrange

from . import GraphMultiHeadAttention
from . import PositionwiseFeedForward

__all__ = ["SpatialGCN", "SpatialAttentionGCN"]


class SpatialGCN(nn.Module):
    """Spatila Graph Convlutional Network for skeleton sequences"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: Tensor,
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        # adjecency matrix
        self.register_buffer("A", A)
        self.mask = nn.Parameter(torch.ones(self.A.size()))

        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(kernel_size, 1),
                    padding=(int((kernel_size - 1) / 2), 0),
                    stride=(stride, 1),
                )
                for _ in range(self.A.size(0))
            ]
        )

        self.relu = nn.ReLU()

    def forward(self, x: Tensor, ret_attn=None) -> Tensor:
        """
        Args:
            x (Tensor): skeleton sequense (N C_in T V)
        Returns:
            Tensor: out.size() = (N, C_out, T, V)
        """
        N, C, T, V = x.size()

        # graph convolution
        out = None
        x = rearrange(x, "N C T V -> (N C T) V")
        for i, a in enumerate(self.A * self.mask):
            xa = torch.mm(x, a)
            xa = rearrange(xa, "(N C T) V -> N C T V", N=N, C=C)

            z = self.conv_list[i](xa)
            out = z + out if out is not None else z

        out = self.relu(out)

        return out if not ret_attn else (out, None)


class SpatialAttentionGCN(SpatialGCN):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: Tensor,
        nhead: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, A=A)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mha = GraphMultiHeadAttention(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_head=nhead,
            d_k=int(self.out_channels / nhead),
            d_v=int(self.out_channels / nhead),
            dropout=dropout,
        )
        self.ffn = PositionwiseFeedForward(
            self.out_channels, self.out_channels * 2, dropout=dropout
        )

        self.relu = nn.ReLU()

        # delete unused conv_list parameters
        delattr(self, "conv_list")

    def forward(self, x: Tensor, ret_attn=None) -> Tensor:
        """
        Args:
            x (Tensor): skeleton sequense: x.size() = (N, C_in, T, V)
        Returns:
            Tensor: out.size() = (N, C_out, T, V)
        """
        N, C, T, V = x.size()

        # attention
        x_attn = rearrange(x, "N C T V -> (N T) V C")
        x_attn, weights = self.mha(x_attn, self.A * self.mask)
        x_attn = self.ffn(x_attn)
        x_attn = rearrange(x_attn, "(N T) V C-> N C T V", T=T)

        if self.in_channels == self.out_channels:
            out = x_attn + x
        else:
            out = x_attn

        out = self.relu(out)

        return out if not ret_attn else (out, weights)
