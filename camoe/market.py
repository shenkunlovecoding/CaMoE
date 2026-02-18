"""
CaMoE v20 Market Mechanism
Top-2 Vickrey Auction + Central Bank/QE + Idle Tax/Depreciation + Asset Velocity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CapitalManager(nn.Module):
    r"""CapitalManager(...) -> None

    管理每层专家资本、税收、折旧、货币政策和资产流速。
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        total_capital: float = 10000.0,
        min_share: float = 0.05,
        tax_threshold: float = 2.0,
        tax_rate: float = 0.1,
        economy: Dict = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.total_capital = float(total_capital)
        self.min_share = float(min_share)
        self.tax_threshold = float(tax_threshold)
        self.tax_rate = float(tax_rate)

        econ = economy or {}
        self.base_compute_floor_ratio = float(econ.get("base_compute_floor_ratio", 0.06))
        self.qe_low_ratio = float(econ.get("qe_low_ratio", 0.85))
        self.qe_high_ratio = float(econ.get("qe_high_ratio", 1.20))
        self.qe_inject_ratio = float(econ.get("qe_inject_ratio", 0.20))
        self.qe_drain_ratio = float(econ.get("qe_drain_ratio", 0.10))
        self.qe_floor_alloc_ratio = float(econ.get("qe_floor_alloc_ratio", 0.70))
        self.idle_threshold = float(econ.get("idle_threshold", 0.01))
        self.idle_tax_rate = float(econ.get("idle_tax_rate", 0.02))
        self.depreciation_rate = float(econ.get("depreciation_rate", 0.001))

        init_cap = self.total_capital / self.num_experts
        self.register_buffer("capitals", torch.ones(num_layers, num_experts) * init_cap)
        self.register_buffer("baseline_losses", torch.ones(num_layers) * 5.0)
        self.register_buffer("selection_ema", torch.zeros(num_layers, num_experts))
        self.register_buffer("asset_flow_ema", torch.zeros(num_layers))
        self.register_buffer("asset_velocity", torch.zeros(num_layers))
        self.register_buffer("last_qe_inject", torch.zeros(num_layers))
        self.register_buffer("last_qe_drain", torch.zeros(num_layers))
        self.register_buffer("last_idle_tax", torch.zeros(num_layers))
        self.register_buffer("last_depreciation", torch.zeros(num_layers))
        self.register_buffer("last_profit_flow", torch.zeros(num_layers))
        self.register_buffer("last_wealth_tax", torch.zeros(num_layers))

    def get_shares(self, layer_idx: int) -> torch.Tensor:
        caps = self.capitals[layer_idx]
        return caps / (caps.sum() + 1e-6)

    def _compute_guarantee_floor(self) -> float:
        floor_by_ratio = self.total_capital * self.base_compute_floor_ratio / self.num_experts
        floor_by_share = self.total_capital * self.min_share / self.num_experts
        return max(floor_by_ratio, floor_by_share)

    def apply_idle_tax_and_depreciation(self, layer_idx: int, winners: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""apply_idle_tax_and_depreciation(layer_idx, winners) -> Dict"""
        caps = self.capitals[layer_idx]
        one_hot = F.one_hot(winners, num_classes=self.num_experts).any(dim=2).float()  # [B, T, E]
        selected_rate = one_hot.mean(dim=(0, 1))
        self.selection_ema[layer_idx] = 0.95 * self.selection_ema[layer_idx] + 0.05 * selected_rate

        idle_mask = self.selection_ema[layer_idx] < self.idle_threshold
        idle_tax = caps * self.idle_tax_rate * idle_mask.to(caps.dtype)
        caps = caps - idle_tax

        depreciation = caps * self.depreciation_rate
        caps = caps - depreciation

        self.capitals[layer_idx] = caps
        self.last_idle_tax[layer_idx] = idle_tax.sum()
        self.last_depreciation[layer_idx] = depreciation.sum()

        return {
            "idle_tax": idle_tax.sum(),
            "depreciation": depreciation.sum(),
        }

    def apply_monetary_policy(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        r"""apply_monetary_policy(layer_idx) -> Dict"""
        caps = self.capitals[layer_idx]
        target = torch.tensor(self.total_capital, device=caps.device, dtype=caps.dtype)
        total = caps.sum()
        floor = torch.tensor(self._compute_guarantee_floor(), device=caps.device, dtype=caps.dtype)

        inject = torch.zeros((), device=caps.device, dtype=caps.dtype)
        drain = torch.zeros((), device=caps.device, dtype=caps.dtype)

        if total < target * self.qe_low_ratio:
            inject = (target - total) * self.qe_inject_ratio
            floor_gap = torch.clamp(floor - caps, min=0.0)
            floor_gap_sum = floor_gap.sum()
            alloc = torch.zeros_like(caps)

            if floor_gap_sum > 0:
                alloc = alloc + (inject * self.qe_floor_alloc_ratio) * (floor_gap / (floor_gap_sum + 1e-6))
            else:
                alloc = alloc + (inject * self.qe_floor_alloc_ratio) / self.num_experts

            alloc = alloc + (inject * (1.0 - self.qe_floor_alloc_ratio)) / self.num_experts
            caps = caps + alloc

        elif total > target * self.qe_high_ratio:
            drain = (total - target) * self.qe_drain_ratio
            caps = caps - drain * (caps / (total + 1e-6))

        caps = torch.clamp(caps, min=floor)
        self.capitals[layer_idx] = caps
        self.last_qe_inject[layer_idx] = inject
        self.last_qe_drain[layer_idx] = drain

        return {
            "qe_inject": inject,
            "qe_drain": drain,
        }

    def get_asset_velocity(self, layer_idx: int) -> torch.Tensor:
        r"""get_asset_velocity(layer_idx) -> Tensor"""
        return self.asset_velocity[layer_idx]

    def update(
        self,
        layer_idx: int,
        winners: torch.Tensor,
        token_losses: torch.Tensor,
        costs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""update(...) -> Dict

        返回本层的资金流动与政策统计。
        """
        with torch.no_grad():
            avg_loss = token_losses.mean()
            self.baseline_losses[layer_idx] = 0.99 * self.baseline_losses[layer_idx] + 0.01 * avg_loss
            baseline = self.baseline_losses[layer_idx]

            caps = self.capitals[layer_idx].clone()

            one_hot = F.one_hot(winners, num_classes=self.num_experts).any(dim=2).float()  # [B, T, E]
            performance = (baseline - token_losses).unsqueeze(-1)  # [B, T, 1]
            expenses = costs.unsqueeze(-1)  # [B, T, 1]
            profit_per_expert = ((performance - expenses) * one_hot).sum(dim=(0, 1))  # [E]

            caps = caps + profit_per_expert

            # 累进税（防垄断）
            avg_cap = caps.mean()
            excess = torch.clamp(caps - avg_cap * self.tax_threshold, min=0.0)
            wealth_tax = excess * self.tax_rate
            caps = caps - wealth_tax
            self.last_wealth_tax[layer_idx] = wealth_tax.sum()

            self.capitals[layer_idx] = caps

            idle_stats = self.apply_idle_tax_and_depreciation(layer_idx, winners)
            qe_stats = self.apply_monetary_policy(layer_idx)

            # 最终保障线
            guarantee_floor = self._compute_guarantee_floor()
            self.capitals[layer_idx] = torch.clamp(self.capitals[layer_idx], min=guarantee_floor)

            # 资产流速：单位 step 的资金周转占比
            flow = profit_per_expert.abs().sum()
            total_cap = self.capitals[layer_idx].sum() + 1e-6
            self.asset_flow_ema[layer_idx] = 0.95 * self.asset_flow_ema[layer_idx] + 0.05 * flow
            self.asset_velocity[layer_idx] = self.asset_flow_ema[layer_idx] / total_cap
            self.last_profit_flow[layer_idx] = flow

            return {
                "profit_flow": flow,
                "wealth_tax": self.last_wealth_tax[layer_idx],
                "idle_tax": idle_stats["idle_tax"],
                "depreciation": idle_stats["depreciation"],
                "qe_inject": qe_stats["qe_inject"],
                "qe_drain": qe_stats["qe_drain"],
                "asset_velocity": self.asset_velocity[layer_idx],
            }


class SparseRouter:
    r"""SparseRouter(noise_std=0.02) -> None

    基于 Top-2 Vickrey 拍卖的稀疏路由器。
    """

    def __init__(self, noise_std: float = 0.02) -> None:
        self.noise_std = noise_std

    def route(
        self,
        confidences: torch.Tensor,
        capital_shares: torch.Tensor,
        difficulty: torch.Tensor,
        critic_subsidy: torch.Tensor = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _E = confidences.shape

        # 1) 计算 bids
        bids = confidences * capital_shares.view(1, 1, -1) * (1.0 + difficulty)
        if critic_subsidy is not None:
            bids = bids + critic_subsidy
        if training:
            bids = bids + torch.randn_like(bids) * self.noise_std

        # 2) Top-3 选举（Top-2 当选，Top-3 定价）
        topk_vals, topk_idxs = torch.topk(bids, 3, dim=-1)
        winners = topk_idxs[:, :, :2]
        price = topk_vals[:, :, 2]

        # 3) Top-2 权重
        top2_bids = topk_vals[:, :, :2]
        weights = F.softmax(top2_bids, dim=-1)

        return winners, weights, price, bids
