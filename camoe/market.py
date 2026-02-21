"""
CaMoE v21 Market Mechanism
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
        self.vc_affinity_threshold = float(econ.get("vc_affinity_threshold", 0.15))
        self.vc_low_cap_ratio = float(econ.get("vc_low_cap_ratio", 0.85))
        self.vc_selected_threshold = float(econ.get("vc_selected_threshold", 0.01))
        self.vc_inject_ratio = float(econ.get("vc_inject_ratio", 0.01))
        self.vc_max_inject_ratio = float(econ.get("vc_max_inject_ratio", 0.12))

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
        self.register_buffer("last_vc_inject", torch.zeros(num_layers))

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
        affinity: torch.Tensor = None,
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

            vc_inject = self.apply_venture_capital(layer_idx, winners, affinity)

            return {
                "profit_flow": flow,
                "wealth_tax": self.last_wealth_tax[layer_idx],
                "idle_tax": idle_stats["idle_tax"],
                "depreciation": idle_stats["depreciation"],
                "qe_inject": qe_stats["qe_inject"],
                "qe_drain": qe_stats["qe_drain"],
                "vc_inject": vc_inject,
                "asset_velocity": self.asset_velocity[layer_idx],
            }

    def apply_venture_capital(
        self,
        layer_idx: int,
        winners: torch.Tensor,
        affinity: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""apply_venture_capital(layer_idx, winners, affinity=None) -> Tensor

        若专家满足「高 affinity + 低资本 + 几乎未被选中」，执行风投注资。
        """
        caps = self.capitals[layer_idx]
        if affinity is None:
            self.last_vc_inject[layer_idx] = 0.0
            return torch.zeros((), device=caps.device, dtype=caps.dtype)

        with torch.no_grad():
            # token 级选择率：Top-2 任一命中即记为被选中
            one_hot = F.one_hot(winners, num_classes=self.num_experts).any(dim=2).float()  # [B,T,E]
            selected_rate = one_hot.mean(dim=(0, 1))  # [E]

            # Critic 偏好强度（只保留正向看好）
            aff_score = F.relu(affinity).mean(dim=(0, 1))  # [E]
            aff_gate = torch.clamp(aff_score - self.vc_affinity_threshold, min=0.0)

            avg_cap = caps.mean()
            low_cap_mask = caps < (avg_cap * self.vc_low_cap_ratio)
            low_select_mask = selected_rate < self.vc_selected_threshold
            eligible = low_cap_mask & low_select_mask & (aff_gate > 0)

            if not eligible.any():
                self.last_vc_inject[layer_idx] = 0.0
                return torch.zeros((), device=caps.device, dtype=caps.dtype)

            # “越穷越值得投”：同等 affinity 下优先低资本专家
            poverty_boost = (avg_cap / (caps + 1e-6)).clamp(min=0.5, max=5.0)
            score = aff_gate * poverty_boost * eligible.to(caps.dtype)
            score_sum = score.sum()
            if score_sum <= 0:
                self.last_vc_inject[layer_idx] = 0.0
                return torch.zeros((), device=caps.device, dtype=caps.dtype)

            budget = torch.tensor(self.total_capital * self.vc_inject_ratio, device=caps.device, dtype=caps.dtype)
            alloc = budget * (score / (score_sum + 1e-6))
            max_each = torch.tensor(
                self.total_capital * self.vc_max_inject_ratio / self.num_experts,
                device=caps.device,
                dtype=caps.dtype,
            )
            alloc = torch.clamp(alloc, min=0.0, max=max_each)
            inject = alloc.sum()

            self.capitals[layer_idx] = caps + alloc
            self.last_vc_inject[layer_idx] = inject
            return inject


class SparseRouter:
    r"""SparseRouter(noise_std=0.02) -> None

    基于 Top-2 Vickrey 拍卖的稀疏路由器。

    说明:
      这里的“稀疏”指输出阶段仅保留 Top-2 experts。
      上游 confidence 仍会对全部 experts 计算（E 较小时开销可控）。
    """

    def __init__(self, noise_std: float = 0.02) -> None:
        self.noise_std = noise_std

    def route(
        self,
        gate_logits: torch.Tensor,
        capital_bias: torch.Tensor = None,
        market_enabled: bool = True,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _E = gate_logits.shape

        # 1) winner 分支：adjusted logits（可含市场偏置）
        adjusted_logits = gate_logits
        if capital_bias is not None:
            adjusted_logits = adjusted_logits + capital_bias
        winner_logits = adjusted_logits
        if training:
            winner_logits = winner_logits + torch.randn_like(winner_logits) * self.noise_std

        topk_vals, topk_idxs = torch.topk(winner_logits, 3, dim=-1)
        winners = topk_idxs[:, :, :2]
        if market_enabled:
            # 市场开启时沿用 Vickrey 风格第二价格（这里用 Top-3）
            costs = topk_vals[:, :, 2]
        else:
            costs = torch.zeros(B, T, device=gate_logits.device, dtype=gate_logits.dtype)

        # 2) weight 分支：纯 Gate 主导（不引入市场梯度）
        top2_gate = torch.gather(gate_logits, dim=-1, index=winners)
        weights = F.softmax(top2_gate, dim=-1)

        # 返回无噪声 adjusted logits 作为调试信号
        return winners, weights, costs, adjusted_logits
