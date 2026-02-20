"""
CaMoE v20 Critic (VC Mode)
支持分阶段奖励缩放、债务重组与破产后参数漂移
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CriticVC(nn.Module):
    r"""CriticVC(n_embd, num_experts, init_capital=10000.0) -> None"""

    def __init__(self, n_embd: int, num_experts: int, init_capital: float = 10000.0) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.init_capital = init_capital

        self.feature = nn.Sequential(
            nn.Linear(n_embd, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        self.difficulty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.affinity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_experts),
        )

        self.register_buffer("capital", torch.tensor(float(init_capital)))
        self.register_buffer("prediction_accuracy", torch.tensor(0.5))
        self.register_buffer("debt", torch.tensor(0.0))
        self.register_buffer("bailout_count", torch.tensor(0.0))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.feature(h)
        difficulty = self.difficulty_head(feat) + 1e-3
        affinity = self.affinity_head(feat)
        return difficulty, affinity

    def apply_to_bids(self, bids: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        return bids + self.subsidy_from_affinity(affinity)

    def subsidy_from_affinity(self, affinity: torch.Tensor) -> torch.Tensor:
        capital_ratio = (self.capital / self.init_capital).clamp(0.1, 2.0)
        return affinity * capital_ratio * 0.05

    def restructure_from_donors(self, donor_state: Dict[str, torch.Tensor], alpha: float = 0.12) -> None:
        r"""restructure_from_donors(donor_state, alpha=0.12) -> None

        将当前 Critic 参数向 donor critic 状态漂移（破产重组）。
        """
        if donor_state is None or len(donor_state) == 0:
            return
        alpha = float(alpha)
        with torch.no_grad():
            for name, p in self.named_parameters():
                donor = donor_state.get(name, None)
                if donor is None:
                    continue
                p.data.lerp_(donor.to(device=p.device, dtype=p.dtype), alpha)

    def settle(
        self,
        affinity: torch.Tensor,
        winners: torch.Tensor,
        token_losses: torch.Tensor,
        baseline: float,
        reward_scale: float = 1.0,
        penalty_scale: float = 1.0,
        critic_bonus_scale: float = 0.0,
        bonus_clip: Tuple[float, float] = (-0.2, 0.4),
        critic_loss_signal: float = 0.0,
        base_commission: float = 1.0,
        dividend_scale: float = 0.0,
        dividend_std_factor: float = 0.5,
    ) -> Dict:
        r"""settle(...) -> Dict

        默认参数与旧版本兼容；新增参数用于 v20 分阶段经济调制。
        """
        with torch.no_grad():
            _B, _T, E = affinity.shape
            amounts = affinity.abs() * 0.05
            directions = torch.sign(affinity)

            total_cost = amounts.sum()
            total_revenue = torch.zeros((), device=affinity.device, dtype=affinity.dtype)
            correct = torch.zeros((), device=affinity.device, dtype=affinity.dtype)
            total = torch.zeros((), device=affinity.device, dtype=affinity.dtype)

            perf = baseline - token_losses
            good = perf > 0

            # 分红：显著优于全场平均时额外加成
            perf_mean = perf.mean()
            perf_std = perf.std(unbiased=False)
            dividend_gate = perf > (perf_mean + dividend_std_factor * perf_std)
            dividend_mul = 1.0 + dividend_scale * torch.clamp(perf - perf_mean, min=0.0)

            for e in range(E):
                won = (winners == e).any(dim=-1)
                lost = ~won

                amt = amounts[:, :, e]
                direction = directions[:, :, e]
                is_long = direction > 0
                is_short = direction < 0

                # 做多：奖励 good+won，弱化 lost
                long_win_good = is_long & won & good
                long_lost = is_long & lost

                # 做空：奖励 short+lost 及 short+won+bad
                short_lost = is_short & lost
                short_win_bad = is_short & won & ~good
                short_win_good = is_short & won & good

                # 奖励项
                rev_long_good = (amt[long_win_good] * 1.5 * reward_scale)
                rev_short_lost = (amt[short_lost] * 1.3 * reward_scale)
                rev_short_win_bad = (amt[short_win_bad] * 1.2 * reward_scale)

                # 惩罚/保底项
                rev_long_lost = (amt[long_lost] * 0.5 * penalty_scale)
                rev_short_win_good = (amt[short_win_good] * 0.2 * penalty_scale)

                if dividend_scale > 0:
                    rev_long_good = rev_long_good * dividend_mul[long_win_good]
                    rev_short_lost = rev_short_lost * dividend_mul[short_lost]
                    rev_short_win_bad = rev_short_win_bad * dividend_mul[short_win_bad]
                    rev_long_lost = rev_long_lost * (1.0 + 0.25 * dividend_gate[long_lost].to(rev_long_lost.dtype))
                    rev_short_win_good = rev_short_win_good * (1.0 + 0.25 * dividend_gate[short_win_good].to(rev_short_win_good.dtype))

                total_revenue = (
                    total_revenue
                    + rev_long_good.sum()
                    + rev_short_lost.sum()
                    + rev_short_win_bad.sum()
                    + rev_long_lost.sum()
                    + rev_short_win_good.sum()
                )

                correct = correct + long_win_good.sum() + short_lost.sum() + short_win_bad.sum()
                total = total + (is_long | is_short).sum()

            # 常规分红制：平时分成降低
            total_revenue = total_revenue * float(base_commission)

            # CriticLoss 奖励：loss 下降越快，奖励越高
            signal = torch.tensor(float(critic_loss_signal), device=affinity.device, dtype=affinity.dtype)
            signal = torch.clamp(signal, min=float(bonus_clip[0]), max=float(bonus_clip[1]))
            bonus_factor = torch.clamp(1.0 + float(critic_bonus_scale) * signal, min=0.1, max=2.0)

            profit = total_revenue * bonus_factor - total_cost
            self.capital = (self.capital + profit).clamp(100.0, self.init_capital * 3)

            if total > 0:
                acc = correct / (total + 1e-6)
                self.prediction_accuracy = 0.95 * self.prediction_accuracy + 0.05 * acc

            return {
                "profit": float(profit.item()),
                "capital": float(self.capital.item()),
                "bonus_factor": float(bonus_factor.item()),
            }
