"""
CaMoE Experts
RWKV FFN + Prefix-Attention Expert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseRWKVFFN(nn.Module):
    r"""SparseRWKVFFN(n_embd, expand=4) -> None

    RWKV 风格 ReLU^2 FFN 专家（无状态版本）。

    Args:
      n_embd (int): 输入/输出通道维度。
      expand (int, optional): FFN 扩展倍率。Default: ``4``。
    """

    def __init__(self, n_embd: int, expand: int = 4) -> None:
        super().__init__()
        hidden = n_embd * expand
        self.key = nn.Linear(n_embd, hidden, bias=False)
        self.value = nn.Linear(hidden, n_embd, bias=False)
        
        self.confidence = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.GELU(), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        
        # Init
        self.value.weight.data.zero_()
        nn.init.orthogonal_(self.key.weight.data, gain=2.0)
        self.forward = torch.compile(self.forward, mode="max-autotune",dynamic=True)
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        r"""get_confidence(x) -> Tensor

        计算 token 级专家置信度。

        Args:
          x (Tensor): 形状 ``[B, T, C]``。

        Returns:
          Tensor: 形状 ``[B, T]`` 的置信度。
        """
        return self.confidence(x).squeeze(-1)
        
    def forward(self, x: torch.Tensor, prefix: torch.Tensor = None) -> torch.Tensor:
        r"""forward(x, prefix=None) -> Tensor

        执行 FFN 前向。

        Args:
          x (Tensor): 形状 ``[N, C]``。
          prefix (Tensor, optional): 占位参数，RWKV 专家不使用。Default: ``None``。

        Returns:
          Tensor: 形状 ``[N, C]`` 的专家输出。
        """
        k = torch.relu(self.key(x)) ** 2
        out = self.value(k)
        return out


class LinearTransformerExpert(nn.Module):
    r"""LinearTransformerExpert(n_embd, n_head) -> None

    使用 Bridge Prefix 作为 K/V 的前缀注意力专家。

    说明:
      名称沿用历史命名，但当前实现使用 ``scaled_dot_product_attention``
      的 softmax attention（非严格线性 attention）。

    Args:
      n_embd (int): 输入/输出维度。
      n_head (int): 注意力头数。
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.o = nn.Linear(n_embd, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_embd)
        
        self.confidence = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.GELU(), 
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        
        # Init: 抑制输出
        nn.init.orthogonal_(self.o.weight, gain=0.1)
        self.forward = torch.compile(self.forward, mode="max-autotune",dynamic=True)

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        r"""get_confidence(x) -> Tensor

        Args:
          x (Tensor): 形状 ``[B, T, C]``。

        Returns:
          Tensor: 形状 ``[B, T]`` 的置信度。
        """
        return self.confidence(x).squeeze(-1)

    def forward(self, x: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        r"""forward(x, prefix) -> Tensor

        Args:
          x (Tensor): 形状 ``[N, C]``，Query 来源。
          prefix (Tensor): 形状 ``[N, P, C]``，由 Bridge 生成的 Key/Value 来源。

        Returns:
          Tensor: 形状 ``[N, C]`` 的专家输出。
        """
        N, C = x.shape
        H, D = self.n_head, self.head_dim
        P = prefix.shape[1]
        
        # Prefix attention (SDPA softmax, not strict linear attention)
        q = self.q(x).reshape(N, 1, H, D)
        k = self.k(prefix).reshape(N, P, H, D)
        v = self.v(prefix).reshape(N, P, H, D)
        
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),  # [N, H, 1, D]
            k.transpose(1, 2),  # [N, H, P, D]
            v.transpose(1, 2),  # [N, H, P, D]
            is_causal=False
        )
        
        out = out.transpose(1, 2).reshape(N, C)
        
        # Gating
        g = torch.sigmoid(self.gate(x))
        out = self.o(out) * g
        
        return out
