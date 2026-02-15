"""
CaMoE v18 Market Mechanism
Top-2 Vickrey Auction with Third-Price Clearing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CapitalManager(nn.Module):
    r"""CapitalManager(num_layers, num_experts, total_capital=10000.0, min_share=0.05, tax_threshold=2.0, tax_rate=0.1) -> None

    管理每一层专家资本的动态更新，用于市场路由的长期激励。

    Args:
      num_layers (int): 模型层数。
      num_experts (int): 每层专家数量。
      total_capital (float, optional): 初始总资本规模。Default: ``10000.0``。
      min_share (float, optional): 每个专家最低资本占比保障系数。Default: ``0.05``。
      tax_threshold (float, optional): 触发累进税的资本倍数阈值。Default: ``2.0``。
      tax_rate (float, optional): 对超额资本征税比例。Default: ``0.1``。
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        total_capital: float = 10000.0,
        min_share: float = 0.05,
        tax_threshold: float = 2.0,
        tax_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.total_capital = total_capital
        self.min_share = min_share
        self.tax_threshold = tax_threshold
        self.tax_rate = tax_rate
        
        init_cap = total_capital / num_experts
        
        self.register_buffer('capitals', torch.ones(num_layers, num_experts) * init_cap)
        self.register_buffer('baseline_losses', torch.ones(num_layers) * 5.0)
    
    def get_shares(self, layer_idx: int) -> torch.Tensor:
        r"""get_shares(layer_idx) -> Tensor

        返回某层专家资本占比，用于路由报价缩放。

        Args:
          layer_idx (int): 层索引。

        Returns:
          Tensor: 形状为 ``[num_experts]`` 的资本占比向量。
        """
        caps = self.capitals[layer_idx]
        return caps / (caps.sum() + 1e-6)
    
    def update(
        self,
        layer_idx: int,
        winners: torch.Tensor,
        token_losses: torch.Tensor,
        costs: torch.Tensor,
    ) -> None:
        r"""update(layer_idx, winners, token_losses, costs) -> None

        根据 token 级收益更新专家资本，并应用税收、保底和总量控制。

        Args:
          layer_idx (int): 当前层索引。
          winners (Tensor): 形状 ``[B, T, 2]``，Top-2 中标专家索引。
          token_losses (Tensor): 形状 ``[B, T]``，token 级损失。
          costs (Tensor): 形状 ``[B, T]``，成交价（第三价格）。
        """
        with torch.no_grad():
            avg_loss = token_losses.mean()
            self.baseline_losses[layer_idx] = 0.99 * self.baseline_losses[layer_idx] + 0.01 * avg_loss
            baseline = self.baseline_losses[layer_idx]
            
            caps = self.capitals[layer_idx].clone()
            
            for e in range(self.num_experts):
                # Top-2 中任一位置选中，即算激活
                mask = (winners == e).any(dim=-1)  # [B, T]
                
                if mask.sum() == 0:
                    continue
                
                real_loss = token_losses[mask]
                expense = costs[mask]
                
                performance = baseline - real_loss
                profit = performance - expense
                caps[e] += profit.sum().item()
            
            # 累进税
            avg_cap = caps.mean()
            for e in range(self.num_experts):
                if caps[e] > avg_cap * self.tax_threshold:
                    excess = caps[e] - avg_cap * self.tax_threshold
                    caps[e] -= excess * self.tax_rate
            
            # 最低保障
            min_cap = self.total_capital * self.min_share / self.num_experts
            caps = torch.clamp(caps, min=min_cap)
            
            # 总量控制
            total = caps.sum()
            if total > self.total_capital * 1.5:
                caps *= 0.95
            elif total < self.total_capital * 0.5:
                caps += self.total_capital * 0.01
            
            self.capitals[layer_idx] = caps


class SparseRouter:
    r"""SparseRouter(noise_std=0.02) -> None

    基于 Top-2 Vickrey 拍卖的稀疏路由器。
    """

    def __init__(self, noise_std: float = 0.02) -> None:
        self.noise_std = noise_std
    
    def route(self, 
              confidences: torch.Tensor,
              capital_shares: torch.Tensor,
              difficulty: torch.Tensor,
              critic_subsidy: torch.Tensor = None,
              training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""route(confidences, capital_shares, difficulty, critic_subsidy=None, training=True) -> Tuple[Tensor, Tensor, Tensor, Tensor]

        执行 Top-3 竞价、Top-2 选举、第三价格清算，并输出 Top-2 混合权重。

        Args:
          confidences (Tensor): 形状 ``[B, T, E]``，专家置信度。
          capital_shares (Tensor): 形状 ``[E]``，专家资本占比。
          difficulty (Tensor): 形状 ``[B, T, 1]``，样本难度。
          critic_subsidy (Tensor, optional): 形状 ``[B, T, E]``，Critic 报价修正。
          training (bool, optional): 是否加探索噪声。Default: ``True``。

        Returns:
          Tuple[Tensor, Tensor, Tensor, Tensor]:
          ``winners`` 形状 ``[B, T, 2]``，``weights`` 形状 ``[B, T, 2]``，
          ``price`` 形状 ``[B, T]``，``bids`` 形状 ``[B, T, E]``。
        """
        B, T, E = confidences.shape
        
        # 1. 计算 Bids
        bids = confidences * capital_shares.view(1, 1, -1) * (1.0 + difficulty)
        
        if critic_subsidy is not None:
            bids = bids + critic_subsidy
        
        if training:
            bids = bids + torch.randn_like(bids) * self.noise_std
        
        # 2. Top-3 Selection (Top-2 当选，Top-3 定价)
        topk_vals, topk_idxs = torch.topk(bids, 3, dim=-1)
        
        # Winners: Top-1 & Top-2
        winners = topk_idxs[:, :, :2]  # [B, T, 2]
        
        # Clearing Price: 3rd Highest Bid
        price = topk_vals[:, :, 2]  # [B, T]
        
        # 3. Mixing Weights
        top2_bids = topk_vals[:, :, :2]
        weights = F.softmax(top2_bids, dim=-1)  # [B, T, 2]
        
        return winners, weights, price, bids
