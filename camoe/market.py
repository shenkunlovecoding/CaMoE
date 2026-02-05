"""
市场机制
稀疏激活版本 - 只让胜者干活
"""

import torch
from typing import Tuple, Dict, List
from torch import nn

class CapitalManager(nn.Module):  # 改成继承 nn.Module
    def __init__(self, num_layers: int, num_experts: int, 
                 total_capital: float = 10000.0, min_share: float = 0.05,
                 tax_threshold: float = 1.5, tax_rate: float = 0.15):
        super().__init__()  # 添加这行
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.total_capital = total_capital
        self.min_share = min_share
        self.tax_threshold = tax_threshold
        self.tax_rate = tax_rate
        
        init_cap = total_capital / num_experts
        
        # [关键改动] 用 register_buffer 代替普通属性
        self.register_buffer('capitals', torch.ones(num_layers, num_experts) * init_cap)
        self.register_buffer('baseline_losses', torch.ones(num_layers) * 5.0)
    
    # 删除 to() 方法，nn.Module 会自动处理
    # def to(self, device): ...  ← 删掉这个
    
    def get_shares(self, layer_idx: int) -> torch.Tensor:
        caps = self.capitals[layer_idx]
        return caps / (caps.sum() + 1e-6)
    
    def update(self, layer_idx: int, winners: torch.Tensor, 
               token_losses: torch.Tensor, costs: torch.Tensor, odds: torch.Tensor):
        with torch.no_grad():
            avg_loss = token_losses.mean()
            self.baseline_losses[layer_idx] = 0.99 * self.baseline_losses[layer_idx] + 0.01 * avg_loss
            baseline = self.baseline_losses[layer_idx]
            
            caps = self.capitals[layer_idx].clone()
            
            for e in range(self.num_experts):
                mask = (winners == e)
                if mask.sum() == 0:
                    continue
                
                real_loss = token_losses[mask]
                expert_odds = odds[mask].squeeze(-1) if odds.dim() > 2 else odds[mask]
                expense = costs[mask]
                
                performance = baseline - real_loss
                revenue = performance * (1.0 + expert_odds)
                profit = revenue - expense
                caps[e] += profit.sum().item()
            
            avg_cap = caps.mean()
            for e in range(self.num_experts):
                if caps[e] > avg_cap * self.tax_threshold:
                    excess = caps[e] - avg_cap * self.tax_threshold
                    caps[e] -= excess * self.tax_rate
            
            min_cap = self.total_capital * self.min_share / self.num_experts
            caps = torch.clamp(caps, min=min_cap)
            
            total = caps.sum()
            if total > self.total_capital * 1.5:
                caps *= 0.95
            elif total < self.total_capital * 0.5:
                caps += self.total_capital * 0.01
            
            self.capitals[layer_idx] = caps


class SparseRouter:
    """
    稀疏路由器
    只让胜者干活，省计算量
    """
    
    def __init__(self, noise_std: float = 0.02):
        self.noise_std = noise_std
    
    def route(self, 
              confidences: torch.Tensor,
              capital_shares: torch.Tensor,
              difficulty: torch.Tensor,
              critic_subsidy: torch.Tensor = None,
              training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算胜者，返回路由信息
        
        confidences: [B, T, E]
        capital_shares: [E]
        difficulty: [B, T, 1]
        
        returns:
            winners: [B, T] - 每个位置的胜者ID
            costs: [B, T] - second price
            bids: [B, T, E] - 完整bid (用于logging)
        """
        B, T, E = confidences.shape
        
        bids = confidences * capital_shares.view(1, 1, -1) * (1.0 + difficulty)
        
        if critic_subsidy is not None:
            bids = bids + critic_subsidy
        
        if training:
            bids = bids + torch.randn_like(bids) * self.noise_std
        
        top2_vals, top2_idxs = torch.topk(bids, 2, dim=-1)
        winners = top2_idxs[:, :, 0]
        costs = top2_vals[:, :, 1]
        
        return winners, costs, bids


class EurekaController:
    def __init__(self, base_epsilon: float = 0.05, conf_threshold: float = 0.3):
        self.base_epsilon = base_epsilon
        self.conf_threshold = conf_threshold
    
    def should_trigger(self, confidences: torch.Tensor, step: int, warmup_steps: int) -> torch.Tensor:
        B, T, E = confidences.shape
        device = confidences.device
        
        if step < warmup_steps:
            eps = self.base_epsilon * 2
        else:
            progress = min((step - warmup_steps) / 20000, 1.0)
            eps = self.base_epsilon * (1 - progress * 0.8)
        
        random_trigger = torch.rand(B, T, device=device) < eps
        max_conf = confidences.max(dim=-1).values
        low_conf = max_conf < self.conf_threshold
        
        return random_trigger | low_conf
    
    def select_underdog(self, capital_shares: torch.Tensor, trigger_mask: torch.Tensor) -> torch.Tensor:
        B, T = trigger_mask.shape
        E = capital_shares.shape[0]
        device = trigger_mask.device
        
        inv_shares = 1.0 / (capital_shares + 0.01)
        probs = inv_shares / inv_shares.sum()
        
        selections = torch.multinomial(probs.expand(B * T, -1), 1).view(B, T)
        selections = torch.where(trigger_mask, selections, torch.full_like(selections, -1))
        
        return selections