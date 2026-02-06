"""
CaMoE v18 Bridge
Linear-State Bridge with Low-Rank Projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UltimateBridge(nn.Module):
    def __init__(self, n_embd: int, max_prefix_len: int = 64, low_rank_dim: int = 64):
        super().__init__()
        self.max_prefix_len = max_prefix_len
        self.n_embd = n_embd
        self.low_rank_dim = low_rank_dim
        
        # 1. 压缩/融合层 [N, 2C] -> [N, 2C]
        self.compressor = nn.Sequential(
            nn.Linear(n_embd * 2, n_embd * 2),
            nn.LayerNorm(n_embd * 2),
            nn.GELU()
        )
        
        # 2. Low-Rank 生成层
        # [N, 2C] -> [N, prefix_len * low_rank_dim]
        self.generator_low = nn.Linear(n_embd * 2, max_prefix_len * self.low_rank_dim)
        
        # 3. 上采样 [low_rank_dim -> n_embd]
        self.upsample = nn.Linear(self.low_rank_dim, n_embd, bias=False)
        
        # 4. 位置编码 [1, prefix_len, n_embd]
        self.pos_emb = nn.Parameter(torch.zeros(1, max_prefix_len, n_embd))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor, rwkv_state: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C] - 当前 Token Embedding
        rwkv_state: [N, C] - RWKV Hidden State (允许梯度回流)
        
        Returns:
            prefix: [N, prefix_len, C]
        """
        N, C = x.shape
        
        # 关键：不再 detach，让梯度流回 RWKV Backbone
        combined = torch.cat([x, rwkv_state], dim=-1)
        
        # 融合特征
        feat = self.compressor(combined)
        
        # 生成 Low-Rank 前缀
        # [N, 2C] -> [N, prefix_len * low_rank_dim] -> [N, prefix_len, low_rank_dim]
        low_feat = self.generator_low(feat).reshape(N, self.max_prefix_len, self.low_rank_dim)
        
        # 上采样回高维空间
        # [N, prefix_len, low_rank_dim] -> [N, prefix_len, C]
        prefix = self.upsample(low_feat)
        
        # 注入位置信息 & Norm
        prefix = prefix + self.pos_emb
        prefix = self.norm(prefix)
        
        return prefix