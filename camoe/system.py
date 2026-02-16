"""
CaMoE v18 主系统 (Final Fix)
Changes:
1. 强制全程开启 Router (use_market=True)，拒绝随机路由。
2. 修复 LinearTransformerExpert 初始化。
3. 保持 Rescale Trick 和 GC。
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple, List
from contextlib import nullcontext
from torch.utils.checkpoint import checkpoint

from .backbone import RWKV7_TimeMix, DeepEmbedAttention, SharedDeepEmbed
from .bridge import UltimateBridge
from .experts import SparseRWKVFFN, LinearTransformerExpert
from .critic import CriticVC
from .market import CapitalManager, SparseRouter

class CaMoE_Block(nn.Module):
    r"""单个 CaMoE Block，包含 TimeMix、DEA 与 Top-2 专家路由。"""
    
    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
        head_size: int,
        config: Dict,
        bridge: nn.Module,
        shared_deep_embed: nn.Module = None,
    ) -> None:
        r"""初始化单层 CaMoE Block。"""
        super().__init__()
        
        self.layer_id = layer_id
        self.num_rwkv = config.get('num_rwkv_experts', 6)
        self.num_trans = config.get('num_trans_experts', 2)
        self.num_experts = self.num_rwkv + self.num_trans
        self.n_embd = n_embd
        self.bridge = bridge
        self.nan_debug = config.get("nan_debug", False)
        use_gc = config.get("use_gradient_checkpoint", True)
        self.checkpoint_att_stage = use_gc and config.get("checkpoint_att_stage", True)
        self.checkpoint_expert_stage = use_gc and config.get("checkpoint_expert_stage", True)
        self.route_no_grad = config.get("route_no_grad", True)
        self.lazy_prefix_union = config.get("lazy_prefix_union", True)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # RWKV-7 TimeMix (Backbone)
        self.att = RWKV7_TimeMix(n_embd, n_layer, layer_id, head_size)
        
        # DeepEmbedAttention (v18.5-test): 与 TimeMix 并行的因果 Attention 分支
        self.use_deep_embed_attention = config.get("use_deep_embed_attention", False)
        vocab_size = config.get("vocab_size", 65536)
        if self.use_deep_embed_attention:
            self.dea = DeepEmbedAttention(
                n_embd=n_embd,
                n_layer=n_layer,
                layer_id=layer_id,
                head_size=head_size,
                vocab_size=vocab_size,
                shared_deep_embed=shared_deep_embed,
                q_dim=config.get("dea_q_dim", 256),
                kv_dim=config.get("dea_kv_dim", 32),
                score_scale=config.get("dea_score_scale", 1024.0),
                cap_scale=config.get("dea_cap_scale", 64.0),
            )
        else:
            self.dea = None
        
        # 专家组
        self.experts = nn.ModuleList()
        
        # RWKV FFN Experts
        for _ in range(self.num_rwkv):
            self.experts.append(SparseRWKVFFN(n_embd))
        
        # Transformer Experts
        n_head = n_embd // head_size
        for _ in range(self.num_trans):
            self.experts.append(LinearTransformerExpert(n_embd, n_head))
        
        # Critic
        self.critic = CriticVC(n_embd, self.num_experts)

    def _assert_finite(self, x: torch.Tensor, name: str, step: int) -> None:
        if (not self.nan_debug) or (x is None):
            return
        if not torch.is_floating_point(x):
            return
        if torch.isfinite(x).all():
            return
        with torch.no_grad():
            bad = ~torch.isfinite(x)
            bad_count = int(bad.sum().item())
            total = x.numel()
            finite_x = x[torch.isfinite(x)]
            if finite_x.numel() > 0:
                vmin = float(finite_x.min().item())
                vmax = float(finite_x.max().item())
            else:
                vmin = float("nan")
                vmax = float("nan")
            print(
                f"❌ NaNDebug-Block | step={step} | block={self.layer_id} | tensor={name} | "
                f"bad={bad_count}/{total} | finite_min={vmin:.6e} | finite_max={vmax:.6e}"
            )
        raise RuntimeError(f"NaN/Inf in block {self.layer_id}, tensor={name}, step={step}")
    
    def _forward_att_stage(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor,
        idx: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""执行 TimeMix(+DEA) 与 ln2，返回 x_after_att/h/v_first/rwkv_state。"""
        self._assert_finite(x, "x_in", step)
        x_ln = self.ln1(x)
        self._assert_finite(x_ln, "x_ln", step)
        att_out, v_first, rwkv_state = self.att(x_ln, v_first)
        self._assert_finite(att_out, "att_out", step)
        self._assert_finite(v_first, "v_first_att", step)
        self._assert_finite(rwkv_state, "rwkv_state", step)
        if self.dea is not None and idx is not None:
            dea_out = self.dea(x_ln, idx)
            self._assert_finite(dea_out, "dea_out", step)
            x_after_att = x + att_out + dea_out
        else:
            x_after_att = x + att_out
        self._assert_finite(x_after_att, "x_after_att", step)
        h = self.ln2(x_after_att)
        self._assert_finite(h, "h_ln2", step)
        return x_after_att, h, v_first, rwkv_state

    def _forward_route_stage(
        self,
        h: torch.Tensor,
        capital_shares: torch.Tensor,
        router: SparseRouter,
        use_market: bool,
        training: bool,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""执行 confidence/critic/router，返回 winners/weights/costs/difficulty/affinity。"""
        route_ctx = torch.no_grad if self.route_no_grad else nullcontext
        with route_ctx():
            route_h = h.detach() if self.route_no_grad else h
            conf_list = [exp.get_confidence(route_h) for exp in self.experts]
            confidences = torch.stack(conf_list, dim=-1)  # [B, T, E]
            self._assert_finite(confidences, "confidences", step)

            if not use_market:
                B, T, _E = confidences.shape
                winners = torch.randint(0, self.num_experts, (B, T, 2), device=h.device)
                weights = torch.ones(B, T, 2, device=h.device) * 0.5
                costs = torch.zeros(B, T, device=h.device)
                difficulty = torch.ones(B, T, 1, device=h.device)
                affinity = torch.zeros(B, T, self.num_experts, device=h.device)
            else:
                difficulty, affinity = self.critic(route_h)
                self._assert_finite(difficulty, "difficulty", step)
                self._assert_finite(affinity, "affinity", step)
                critic_subsidy = self.critic.subsidy_from_affinity(affinity)
                self._assert_finite(critic_subsidy, "critic_subsidy", step)
                winners, weights, costs, bids = router.route(
                    confidences, capital_shares, difficulty, critic_subsidy, training
                )
                self._assert_finite(weights, "weights", step)
                self._assert_finite(costs, "costs", step)
                self._assert_finite(bids, "bids", step)

        return winners.detach(), weights.detach(), costs.detach(), difficulty.detach(), affinity.detach()

    def _build_trans_prefix_union(
        self,
        h: torch.Tensor,
        rwkv_state: torch.Tensor,
        winners: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""仅为 Transformer 命中 token 构建 prefix；返回 prefix_union 与索引映射。"""
        B, T, C = h.shape
        flat_bt = B * T
        flat_h = h.reshape(flat_bt, C)
        flat_state = rwkv_state.reshape(flat_bt, C)

        if not self.lazy_prefix_union:
            bridge_prefix = self.bridge(flat_h, flat_state)  # [B*T, P, C]
            self._assert_finite(bridge_prefix, "bridge_prefix_full", step)
            prefix_indices = torch.arange(flat_bt, device=h.device, dtype=torch.long)
            return bridge_prefix, prefix_indices

        trans_mask = (winners[:, :, 0] >= self.num_rwkv) | (winners[:, :, 1] >= self.num_rwkv)
        flat_mask = trans_mask.reshape(-1)
        prefix_indices = torch.full((flat_bt,), -1, device=h.device, dtype=torch.long)

        if not flat_mask.any():
            empty_prefix = torch.empty(
                0,
                self.bridge.max_prefix_len,
                C,
                device=h.device,
                dtype=h.dtype,
            )
            return empty_prefix, prefix_indices

        prefix_union = self.bridge(flat_h[flat_mask], flat_state[flat_mask])  # [N_u, P, C]
        self._assert_finite(prefix_union, "bridge_prefix_union", step)
        prefix_indices[flat_mask] = torch.arange(prefix_union.shape[0], device=h.device, dtype=torch.long)
        return prefix_union, prefix_indices

    def _forward_expert_stage(
        self,
        x_after_att: torch.Tensor,
        h: torch.Tensor,
        rwkv_state: torch.Tensor,
        winners: torch.Tensor,
        weights: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        r"""执行 Top-2 专家混合并返回 block 输出。"""
        B, T, C = h.shape
        prefix_union, prefix_indices = self._build_trans_prefix_union(h, rwkv_state, winners, step)
        final_out = torch.zeros_like(h)  # [B, T, C]

        for rank in range(2):
            rank_winners = winners[:, :, rank]  # [B, T]
            rank_weights = weights[:, :, rank].unsqueeze(-1)  # [B, T, 1]

            for e in range(self.num_experts):
                mask = (rank_winners == e)  # [B, T]
                if not mask.any():
                    continue

                selected_h = h[mask]  # [N, C]
                selected_weights = rank_weights[mask]  # [N, 1]

                if e >= self.num_rwkv:
                    flat_mask = mask.reshape(-1)
                    if self.lazy_prefix_union:
                        sel_idx = prefix_indices[flat_mask]
                        valid = sel_idx >= 0
                        if not valid.any():
                            continue
                        expert_out = torch.zeros_like(selected_h)
                        expert_out[valid] = self.experts[e](selected_h[valid], prefix_union[sel_idx[valid]])
                    else:
                        expert_out = self.experts[e](selected_h, prefix_union[flat_mask])
                else:
                    expert_out = self.experts[e](selected_h, None)
                self._assert_finite(expert_out, f"expert_out_e{e}", step)

                weighted_out = expert_out * selected_weights
                self._assert_finite(weighted_out, f"weighted_out_e{e}", step)
                final_out[mask] += weighted_out

        self._assert_finite(final_out, "final_out", step)
        x_out = x_after_att + final_out
        self._assert_finite(x_out, "x_out", step)
        return x_out

    def forward(self, 
                x: torch.Tensor, 
                v_first: torch.Tensor,
                capital_shares: torch.Tensor,
                router: SparseRouter,
                step: int,
                warmup_steps: int,
                use_market: bool = True,
                training: bool = True,
                idx: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        r"""forward(x, v_first, capital_shares, router, step, warmup_steps, use_market=True, training=True, idx=None) -> Tuple[Tensor, Tensor, Dict]"""
        del warmup_steps

        if self.training and self.checkpoint_att_stage:
            x_after_att, h, v_first, rwkv_state = checkpoint(
                lambda xx, vv, ii: self._forward_att_stage(xx, vv, ii, step),
                x,
                v_first,
                idx,
                use_reentrant=False,
            )
        else:
            x_after_att, h, v_first, rwkv_state = self._forward_att_stage(x, v_first, idx, step)

        winners, weights, costs, difficulty, affinity = self._forward_route_stage(
            h, capital_shares, router, use_market, training, step
        )

        if self.training and self.checkpoint_expert_stage:
            x = checkpoint(
                lambda x0, h0, s0, w0, wt0: self._forward_expert_stage(x0, h0, s0, w0, wt0, step),
                x_after_att,
                h,
                rwkv_state,
                winners,
                weights,
                use_reentrant=False,
            )
        else:
            x = self._forward_expert_stage(x_after_att, h, rwkv_state, winners, weights, step)

        info = {
            "winners": winners,
            "costs": costs,
            "difficulty": difficulty,
            "affinity": affinity,
        }
        return x, v_first, info


class CaMoE_System(nn.Module):
    r"""CaMoE 主系统，封装多层 Block、市场状态与损失计算。"""

    def __init__(self, config: Dict) -> None:
        r"""初始化系统级模块与共享组件。"""
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        self.n_layer = config['n_layer']
        self.vocab_size = config['vocab_size']
        self.use_gradient_checkpoint = config.get("use_gradient_checkpoint", True)
        self.nan_debug = config.get("nan_debug", False)
        
        self.num_rwkv_experts = config.get('num_rwkv_experts', 6)
        self.num_trans_experts = config.get('num_trans_experts', 2)
        self.num_experts = self.num_rwkv_experts + self.num_trans_experts
        
        # Embedding
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)

        # Shared DeepEmbed table (optional, recommended for VRAM efficiency)
        self.deep_embed = None
        if config.get("use_deep_embed_attention", False) and config.get("use_shared_deep_embed", True):
            self.deep_embed = SharedDeepEmbed(
                vocab_size=self.vocab_size,
                k_dim=min(config.get("dea_q_dim", 256), self.n_embd),
                v_dim=self.n_embd,
            )
        
        # 共享 Bridge
        self.bridge = UltimateBridge(
            self.n_embd, 
            config.get('prefix_len', 64),
            config.get('low_rank_dim', 64)
        )
        
        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_layer):
            self.blocks.append(CaMoE_Block(
                self.n_embd,
                self.n_layer,
                i,
                config['head_size'],
                config,
                bridge=self.bridge,
                shared_deep_embed=self.deep_embed,
            ))
        
        self.ln_out = nn.LayerNorm(self.n_embd)
        
        # Head (可选 Tied Embedding)
        if config.get('tied_embeddings', False):
            self.head = None  # 使用 emb.weight
        else:
            self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Market
        self.capital_manager = CapitalManager(
            self.n_layer, self.num_experts,
            total_capital=config.get('total_capital', 10000.0),
            min_share=config.get('min_capital_share', 0.05),
            tax_threshold=config.get('tax_threshold', 2.0),
            tax_rate=config.get('tax_rate', 0.1)
        )
        
        self.router = SparseRouter()

    def _assert_finite(self, x: torch.Tensor, name: str, step: int, layer_id: int = -1) -> None:
        r"""_assert_finite(x, name, step, layer_id=-1) -> None

        在调试模式下校验张量数值合法性，出现 NaN/Inf 立即抛错并输出定位信息。
        """
        if (not self.nan_debug) or (x is None):
            return
        if not torch.is_floating_point(x):
            return
        if torch.isfinite(x).all():
            return

        with torch.no_grad():
            bad = ~torch.isfinite(x)
            bad_count = int(bad.sum().item())
            total = x.numel()
            finite_x = x[torch.isfinite(x)]
            if finite_x.numel() > 0:
                vmin = float(finite_x.min().item())
                vmax = float(finite_x.max().item())
            else:
                vmin = float("nan")
                vmax = float("nan")
            print(
                f"❌ NaNDebug | step={step} | layer={layer_id} | tensor={name} | "
                f"bad={bad_count}/{total} | finite_min={vmin:.6e} | finite_max={vmax:.6e}"
            )
        raise RuntimeError(f"NaN/Inf detected at step={step}, layer={layer_id}, tensor={name}")
    
    def forward(self, idx: torch.Tensor, step: int = 0, 
                phase: str = "normal") -> Tuple[torch.Tensor, Dict]:
        r"""forward(idx, step=0, phase="normal") -> Tuple[Tensor, Dict]

        执行整网前向并收集各层路由信息。

        Args:
          idx (Tensor): 形状 ``[B, T]`` 的 token id。
          step (int, optional): 当前步数。Default: ``0``。
          phase (str, optional): 训练阶段标签。Default: ``"normal"``。

        Returns:
          Tuple[Tensor, Dict]: ``logits`` 与各层 ``info``。
        """
        x = self.emb(idx)
        self._assert_finite(x, "emb_out", step, -1)
        v_first = None
        
        # [CRITICAL FIX] 始终开启 Market Routing
        # 即使在 Prewarm/Warmup，我们也需要 Router 选出最好的专家，让专家获得正确的梯度
        # 资本的更新 (Update) 由 train.py 控制，这里只管路由 (Selection)
        use_market = True 
        
        all_info = {
            "winners": [], "costs": [], "difficulties": [], "affinities": []
        }
        warmup_steps = self.config.get('warmup_steps', 2000)

        for i, block in enumerate(self.blocks):
            shares = self.capital_manager.get_shares(i)
            x, v_first, info = block(
                x, v_first, shares, self.router,
                step, warmup_steps, use_market, self.training, idx
            )

            self._assert_finite(x, "block_out", step, i)
            self._assert_finite(v_first, "v_first", step, i)
            self._assert_finite(info["costs"], "costs", step, i)
            self._assert_finite(info["difficulty"], "difficulty", step, i)
            self._assert_finite(info["affinity"], "affinity", step, i)
            
            all_info["winners"].append(info["winners"].detach())
            all_info["costs"].append(info["costs"].detach())
            all_info["difficulties"].append(info["difficulty"].detach())
            all_info["affinities"].append(info["affinity"].detach())
        
        x = self.ln_out(x)
        self._assert_finite(x, "ln_out", step, self.n_layer)
        
        # Output (Tied Embedding Rescale Trick)
        if self.head is not None:
            logits = self.head(x)
        else:
            # Tied embedding 需缩放，避免 logits 幅度过大导致 CE 量纲异常
            x = x * (self.n_embd ** -0.5)
            logits = F.linear(x, self.emb.weight)
        self._assert_finite(logits, "logits", step, self.n_layer)
        
        return logits, all_info
    
    def compute_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        all_info: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        r"""compute_losses(logits, targets, all_info) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]

        计算主损失、token 级损失以及 Critic 损失。

        Args:
          logits (Tensor): 形状 ``[B, T, V]``。
          targets (Tensor): 形状 ``[B, T]``。
          all_info (Dict): 各层难度/路由信息。

        Returns:
          Tuple[Tensor, Tensor, Tensor, Tensor, float]:
          ``total_loss``、``token_losses``、``main_loss``、``critic_loss``、``bridge_loss``。
        """
        if self.config.get("stabilize_logits", False):
            # 训练稳定性保护：避免上游极端值导致 CE 直接 NaN/Inf
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        B, T = targets.shape
        
        # Main Loss
        main_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        
        # Token Losses (for Market Update)
        with torch.no_grad():
            token_losses = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                reduction='none',
                ignore_index=-100,
            ).reshape(B, T)
            if self.config.get("stabilize_logits", False):
                token_losses = torch.nan_to_num(token_losses, nan=0.0, posinf=100.0, neginf=0.0)
            # ignore_index 位置本身为 0 loss，这里再显式归零，避免后续市场更新误用
            token_losses = token_losses.masked_fill(targets.eq(-100), 0.0)
        
        # Critic Loss
        critic_loss = 0.0
        for i, diff in enumerate(all_info.get("difficulties", [])):
            baseline = self.capital_manager.baseline_losses[i]
            target = F.relu(token_losses - baseline)
            critic_loss += F.smooth_l1_loss(diff.squeeze(-1), target)
        
        if len(all_info.get("difficulties", [])) > 0:
            critic_loss /= len(all_info["difficulties"])

        total_loss = main_loss + 0.1 * critic_loss
        bridge_loss = 0.0 # No longer used
        
        return total_loss, token_losses, main_loss, critic_loss, bridge_loss
    
    def update_market(self, all_info: Dict, token_losses: torch.Tensor, step: int) -> None:
        r"""update_market(all_info, token_losses, step) -> None

        根据 token 级损失结算市场状态与 Critic 资本。

        Args:
          all_info (Dict): 前向收集的市场信息。
          token_losses (Tensor): 形状 ``[B, T]``。
          step (int): 当前训练步。
        """
        with torch.no_grad():
            for i in range(self.n_layer):
                if i >= len(all_info.get("winners", [])): 
                    continue
                
                self.capital_manager.update(
                    i, all_info["winners"][i], token_losses, all_info["costs"][i]
                )
                
                baseline = self.capital_manager.baseline_losses[i].item()
                self.blocks[i].critic.settle(
                    all_info["affinities"][i], all_info["winners"][i],
                    token_losses, baseline
                )

                # Bailout logic
                if self.blocks[i].critic.capital < 200:
                    self.blocks[i].critic.capital.fill_(2000.0)
                    if step % 100 == 0:
                        print(f"🏛️  Layer {i}: Critic Bailout (Step {step})")
    
    def log_market_health(self) -> Dict:
        r"""log_market_health() -> Dict

        汇总所有层的市场健康指标。

        Returns:
          Dict: 包含 RWKV/Transformer 份额、Gini、Critic 资本等指标。
        """
        metrics = {}
        for i in range(self.n_layer):
            caps = self.capital_manager.capitals[i]
            total_cap = caps.sum() + 1e-6
            
            rwkv_share = caps[:self.blocks[i].num_rwkv].sum() / total_cap * 100
            trans_share = caps[self.blocks[i].num_rwkv:].sum() / total_cap * 100
            
            sorted_caps, _ = torch.sort(caps)
            n = self.num_experts
            idx = torch.arange(1, n + 1, device=caps.device, dtype=caps.dtype)
            gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * total_cap + 1e-6)
            
            metrics[f"L{i}/TransShare"] = trans_share.item()
            metrics[f"L{i}/RWKVShare"] = rwkv_share.item()
            metrics[f"L{i}/Gini"] = gini.item()
            metrics[f"L{i}/CriticCap"] = self.blocks[i].critic.capital.item()
        
        return metrics
