import torch
from torch import nn, Tensor
from einops import rearrange
import math

from .. import gnn


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1, T, C
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class STGraphormer(nn.Module):
    """
    Spatial temporal graph convolutional networks for skeleton-based action recognition.
    """

    def __init__(
        self,
        in_channels: int,
        num_class: int,
        s_nhead: int,
        t_nhead: int,
        graph_args: dict,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.5,
    ):
        super(STGraphormer, self).__init__()
        # graph
        graph = gnn.Graph(**graph_args)
        A = torch.tensor(graph.A, dtype=torch.float, requires_grad=False)

        # graph for attention
        graph_args_attn = {k: v for k, v in graph_args.items() if k != "strategy"}
        graph2 = gnn.Graph(strategy="DAD", **graph_args_attn)
        A_attn = torch.tensor(graph2.A, dtype=torch.float, requires_grad=False) / 3

        # layers
        self.data_bn = nn.BatchNorm1d(in_channels * graph_args.num_node)

        self.gcn_layers = nn.ModuleList(
            (
                gnn.SpatialGCN(in_channels, 64, A),
                gnn.SpatialGCN(64, 64, A),
                gnn.SpatialGCN(64, 128, A),
                gnn.SpatialGCN(128, 128, A),
                gnn.SpatialAttentionGCN(128, 256, A_attn, s_nhead, dropout),
                gnn.SpatialAttentionGCN(256, 256, A_attn, s_nhead, dropout),
            )
        )

        # tail
        self.pos_encoder = PositionalEncoding(
            d_model=256,
            max_len=2500,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=t_nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Linear(in_features=256, out_features=num_class)

    def forward(self, x: Tensor, ret_attn: bool = None):
        N, C, T, V = x.size()  # N, C, T,V
        # data bn
        x = rearrange(x, "N C T V -> N (V C) T")
        x = self.data_bn(x)
        x = rearrange(x, "N (V C) T -> N C T V", V=V)

        # sptaio-temporal transformer
        gcn_s_attn = []
        for _, m in enumerate(self.gcn_layers):
            if not ret_attn:
                x = m(x, ret_attn)
            else:
                x, _s_attn = m(x, ret_attn)
                gcn_s_attn.append(_s_attn)

        # pooling   NCTV
        x = rearrange(x, "N C T V -> N T C V")
        x = torch.mean(x, dim=-1)  # N T C

        # transformer
        x = self.pos_encoder(x)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_encoder(x)

        # classifier
        scores = self.classifier(x[:, 0])  # input cls token

        if not ret_attn:
            return scores
        else:
            # TODO: extract and return temporal attention weights
            return scores, gcn_s_attn
