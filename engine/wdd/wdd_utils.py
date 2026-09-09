
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple, List, Optional

from ..wdd.utils import get_activation

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ZISGate(nn.Module):

    def __init__(self, d_model, reduction=4, use_rmsnorm=True):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model // reduction),
            nn.Mish(),
            nn.Linear(d_model // reduction, 2 * d_model),
            nn.Sigmoid()
        )
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):

        init.xavier_uniform_(self.gate_proj[0].weight)
        init.constant_(self.gate_proj[0].bias, 0)

        init.constant_(self.gate_proj[2].weight, 0)
        init.constant_(self.gate_proj[2].bias, 0)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = self.gate_proj(gate_input)
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        output = output * self.scale
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'

class MishFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.mish(x1) * x2
        return self.w3(hidden)