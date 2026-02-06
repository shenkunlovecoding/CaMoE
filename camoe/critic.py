"""
CaMoE v18 Critic (VC Mode)
做多/做空实现，适配 Top-2
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class CriticVC(nn.Module):
    def __init__(self, n_embd: int, num_experts: int, init_capital: float = 10000.0):
        super().__init__()
        self.num_experts = num_experts
        self.init_capital = init_capital
        
        self.feature = nn.Sequential(
            nn.Linear(n_embd, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # 难度预测
        self.difficulty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # 专家适合度 (正=做多, 负=做空)
        self.affinity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_experts)
        )
        
        self.register_buffer('capital', torch.tensor(init_capital))
        self.register_buffer('prediction_accuracy', torch.tensor(0.5))
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature(h)
        difficulty = self.difficulty_head(feat) + 1e-3
        affinity = self.affinity_head(feat)
        return difficulty, affinity
    
    def apply_to_bids(self, bids: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        capital_ratio = (self.capital / self.init_capital).clamp(0.1, 2.0)
        modification = affinity * capital_ratio * 0.05
        return bids + modification
    
    def settle(self, affinity: torch.Tensor, winners: torch.Tensor, 
               token_losses: torch.Tensor, baseline: float) -> Dict:
        """
        winners: [B, T, 2] - Top-2 中标者
        """
        with torch.no_grad():
            B, T, E = affinity.shape
            
            amounts = affinity.abs() * 0.05
            directions = torch.sign(affinity)
            
            total_cost = amounts.sum().item()
            total_revenue = 0.0
            correct = 0
            total = 0
            
            for e in range(E):
                # Top-2 中任一位置选中，即算 won
                won = (winners == e).any(dim=-1)  # [B, T]
                lost = ~won
                
                amt = amounts[:, :, e]
                dir = directions[:, :, e]
                is_long = (dir > 0)
                is_short = (dir < 0)
                
                perf = baseline - token_losses
                good = (perf > 0)
                
                # 做多收益
                long_win_good = is_long & won & good
                total_revenue += (amt[long_win_good] * 1.5).sum().item()
                correct += long_win_good.sum().item()
                
                long_lose = is_long & lost
                total_revenue += (amt[long_lose] * 0.5).sum().item()
                
                # 做空收益
                short_lose = is_short & lost
                total_revenue += (amt[short_lose] * 1.3).sum().item()
                correct += short_lose.sum().item()
                
                short_win_bad = is_short & won & ~good
                total_revenue += (amt[short_win_bad] * 1.2).sum().item()
                correct += short_win_bad.sum().item()
                
                short_win_good = is_short & won & good
                total_revenue += (amt[short_win_good] * 0.2).sum().item()
                
                total += (is_long | is_short).sum().item()
            
            profit = total_revenue - total_cost
            self.capital = (self.capital + profit).clamp(100.0, self.init_capital * 3)
            
            if total > 0:
                acc = correct / total
                self.prediction_accuracy = 0.95 * self.prediction_accuracy + 0.05 * acc
            
            return {"profit": profit, "capital": self.capital.item()}