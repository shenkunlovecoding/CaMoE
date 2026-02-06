"""
CaMoE v18 Experts
RWKV FFN + Linear Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseRWKVFFN(nn.Module):
    """
    RWKV FFN 专家 (无状态版)
    """
    def __init__(self, n_embd: int, expand: int = 4):
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
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [B, T] 的置信度"""
        return self.confidence(x).squeeze(-1)
        
    def forward(self, x: torch.Tensor, prefix: torch.Tensor = None) -> torch.Tensor:
        """
        x: [N, C]
        prefix: 忽略 (RWKV 不需要 Bridge)
        """
        k = torch.relu(self.key(x)) ** 2
        out = self.value(k)
        return out


class LinearTransformerExpert(nn.Module):
    """
    Linear Transformer 专家
    使用 Bridge 生成的 Prefix 作为 K/V
    """
    def __init__(self, n_embd: int, n_head: int):
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

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [B, T] 的置信度"""
        return self.confidence(x).squeeze(-1)

    def forward(self, x: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C] - Query 来源
        prefix: [N, P, C] - Key/Value 来源 (由 Bridge 生成)
        """
        N, C = x.shape
        H, D = self.n_head, self.head_dim
        P = prefix.shape[1]
        
        # Linear Attention
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