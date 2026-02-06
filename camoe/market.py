"""
CaMoE v18 Market Mechanism
Top-2 Vickrey Auction with Third-Price Clearing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class CapitalManager(nn.Module):
    def __init__(self, num_layers: int, num_experts: int, 
                 total_capital: float = 10000.0, min_share: float = 0.05,
                 tax_threshold: float = 2.0, tax_rate: float = 0.1):
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
        caps = self.capitals[layer_idx]
        return caps / (caps.sum() + 1e-6)
    
    def update(self, layer_idx: int, winners: torch.Tensor, 
               token_losses: torch.Tensor, costs: torch.Tensor):
        """
        winners: [B, T, 2] - Top-2 中标者
        token_losses: [B, T]
        costs: [B, T] - 成交价 (Third-price)
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
    """
    v18: Top-2 Vickrey Auction Router
    """
    def __init__(self, noise_std: float = 0.02):
        self.noise_std = noise_std
    
    def route(self, 
              confidences: torch.Tensor,
              capital_shares: torch.Tensor,
              difficulty: torch.Tensor,
              critic_subsidy: torch.Tensor = None,
              training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            confidences: [B, T, E]
            capital_shares: [E]
            difficulty: [B, T, 1]
            critic_subsidy: [B, T, E] (可选)
        
        Returns:
            winners: [B, T, 2] - Top-2 专家索引
            weights: [B, T, 2] - 混合权重 (Softmax)
            price: [B, T] - 成交价 (Third-price)
            bids: [B, T, E] - 完整报价
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