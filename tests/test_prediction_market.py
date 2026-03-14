from __future__ import annotations

import unittest

import torch

from camoe.auction import PredictionMarketRouter
from camoe.block import CaMoE_Block
from camoe.capital import MarketStateManager
from camoe.config import CaMoEConfig
from camoe.expert_base import BaseExpert
from camoe.expert_critic import RewardCritic
from camoe.expert_rosa import ROSAExpert
from camoe.model import CaMoE_Model
from camoe.rosa_soft_adapter import rosa_soft


class DummyTimeMixExpert(BaseExpert):
    def __init__(self, dim: int, offset: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.offset = torch.nn.Parameter(torch.full((dim,), float(offset)))

    @property
    def expert_type(self) -> str:
        return "timemix"

    def forward(self, x: torch.Tensor, v_first: torch.Tensor | None = None, **ctx):
        del ctx
        out = x + self.offset.view(*([1] * (x.ndim - 1)), -1)
        next_v = torch.zeros_like(x) if v_first is None else v_first
        state = out.clone()
        return out, next_v, state


class DummyExpert(BaseExpert):
    def __init__(self, dim: int, label: str, offset: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.offset = torch.nn.Parameter(torch.full((dim,), float(offset)))

    @property
    def expert_type(self) -> str:
        return self.label

    def forward(self, x: torch.Tensor, **ctx) -> torch.Tensor:
        del ctx
        view_shape = [1] * x.ndim
        view_shape[-1] = -1
        return x + self.offset.view(*view_shape)


class DummyModel(CaMoE_Model):
    def _build_block(self, layer_idx: int) -> CaMoE_Block:
        del layer_idx
        dim = self.config.dim
        return CaMoE_Block(
            timemix_expert=DummyTimeMixExpert(
                dim=dim,
                offset=0.1,
                capital_init=self.config.expert_capital_init,
                capital_floor=self.config.capital_floor,
                capital_ceiling=self.config.capital_ceiling,
            ),
            rosa_expert=DummyExpert(
                dim=dim,
                label="rosa",
                offset=-0.1,
                capital_init=self.config.expert_capital_init,
                capital_floor=self.config.capital_floor,
                capital_ceiling=self.config.capital_ceiling,
            ),
            ffn_experts=[
                DummyExpert(
                    dim=dim,
                    label="ffn_a",
                    offset=0.2,
                    capital_init=self.config.expert_capital_init,
                    capital_floor=self.config.capital_floor,
                    capital_ceiling=self.config.capital_ceiling,
                ),
                DummyExpert(
                    dim=dim,
                    label="ffn_b",
                    offset=-0.2,
                    capital_init=self.config.expert_capital_init,
                    capital_floor=self.config.capital_floor,
                    capital_ceiling=self.config.capital_ceiling,
                ),
            ],
            sequence_reward_critic=RewardCritic(dim=dim, n_routable=2, hidden_dim=8),
            ffn_reward_critic=RewardCritic(dim=dim, n_routable=2, hidden_dim=8),
            routing_noise_std=0.0,
            use_gradient_checkpointing=False,
            use_routing_ste=True,
            ste_temperature=1.0,
        )


class PredictionMarketTests(unittest.TestCase):
    def test_rosa_soft_adapter_smoke_cpu(self) -> None:
        q = torch.randn(1, 6, 8, requires_grad=True)
        k = torch.randn(1, 6, 8, requires_grad=True)
        v = torch.randn(1, 6, 8, requires_grad=True)

        for mode in ("soft", "sufa", "scan"):
            out = rosa_soft(
                q,
                k,
                v,
                bits_per_symbol=4,
                mode=mode,
                truncation_length=4,
            )
            self.assertEqual(tuple(out.shape), (1, 6, 8))
            loss = out.sum()
            loss.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(out).all().item())
            self.assertTrue(torch.isfinite(q.grad).all().item())
            q.grad.zero_()
            k.grad.zero_()
            v.grad.zero_()

    def test_rosa_expert_sufa_backend_smoke(self) -> None:
        expert = ROSAExpert(
            dim=8,
            slim_heads=2,
            bits_per_symbol=4,
            backend="sufa",
            truncation_length=4,
            sequence_length=6,
        )
        x = torch.randn(2, 6, 8, requires_grad=True)
        out = expert(x)
        self.assertEqual(tuple(out.shape), (2, 6, 8))
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_rosa_symbol_language_summary_smoke(self) -> None:
        expert = ROSAExpert(
            dim=8,
            slim_heads=2,
            bits_per_symbol=4,
            backend="sufa",
            truncation_length=4,
            sequence_length=6,
        )
        x = torch.randn(2, 6, 8)
        valid_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        summary = expert.summarize_symbol_language(x, valid_mask=valid_mask, max_tokens=4, max_heads=1)
        self.assertIn("q_vocab_mean", summary)
        self.assertIn("sample0_q", summary)
        self.assertIn("top_q_symbols", summary)
        self.assertGreaterEqual(float(summary["q_vocab_mean"]), 1.0)

    def test_price_normalization_and_floor(self) -> None:
        q = torch.tensor([5.0, -1.0, -4.0])
        prices = PredictionMarketRouter.compute_prices(q, temperature=1.0, liquidity_floor=0.12)
        self.assertAlmostEqual(float(prices.sum().item()), 1.0, places=6)
        self.assertTrue(torch.all(prices >= 0.12 / 3 - 1e-6).item())

    def test_routing_uses_expected_profit(self) -> None:
        router = PredictionMarketRouter(noise_std=0.0)
        capitals = torch.tensor([20.0, 10.0])
        q = torch.log(torch.tensor([0.9, 0.1]))
        pred_reward = torch.tensor([[[0.91, 0.60]]])
        result = router(
            expert_capitals=capitals,
            q=q,
            reward_logits=torch.logit(pred_reward),
            bet_fraction=0.05,
            price_temperature=1.0,
            liquidity_floor=0.0,
            training=False,
        )
        self.assertEqual(int(result["winners"][0, 0].item()), 1)

    def test_force_winner_overrides_market_choice(self) -> None:
        router = PredictionMarketRouter(noise_std=0.0)
        capitals = torch.tensor([20.0, 10.0])
        q = torch.log(torch.tensor([0.9, 0.1]))
        pred_reward = torch.tensor([[[0.91, 0.60]]])
        result = router(
            expert_capitals=capitals,
            q=q,
            reward_logits=torch.logit(pred_reward),
            bet_fraction=0.05,
            price_temperature=1.0,
            liquidity_floor=0.0,
            force_winner=0,
            training=True,
        )
        self.assertEqual(int(result["winners"][0, 0].item()), 0)

    def test_settlement_updates_only_active_winner_and_keeps_inactive_constant(self) -> None:
        manager = MarketStateManager(
            n_layers=1,
            n_experts_per_layer=3,
            capital_init=100.0,
            capital_floor=1.0,
            price_lr=0.1,
            liquidity_floor=0.0,
            reward_scale=5.0,
        )
        manager.loss_ema[0] = 2.0
        old_caps = manager.capitals[0].clone()
        old_q = manager.q[0].clone()

        result = manager.settle_layer(
            layer_idx=0,
            winners=torch.tensor([[0, 0]]),
            token_loss=torch.tensor([[0.1, 0.1]]),
            prices=torch.tensor([0.5, 0.3, 0.2]),
            shares=torch.tensor([10.0, 5.0, 2.0]),
            token_weight=torch.ones(1, 2),
            update_state=True,
        )

        self.assertGreater(float(result["profit"][0].item()), 0.0)
        self.assertAlmostEqual(float(result["profit"][1].item()), 0.0, places=6)
        self.assertAlmostEqual(float(result["profit"][2].item()), 0.0, places=6)
        self.assertGreater(float(manager.capitals[0, 0].item()), float(old_caps[0].item()))
        self.assertAlmostEqual(float(manager.capitals[0, 1].item()), float(old_caps[1].item()), places=6)
        self.assertAlmostEqual(float(manager.capitals[0, 2].item()), float(old_caps[2].item()), places=6)
        self.assertGreater(float(manager.q[0, 0].item()), float(old_q[0].item()))
        self.assertAlmostEqual(float(manager.q[0, 1].item()), float(old_q[1].item()), places=6)

    def test_batch_invariance_under_duplicate_tokens(self) -> None:
        manager_single = MarketStateManager(
            n_layers=1,
            n_experts_per_layer=2,
            capital_init=50.0,
            capital_floor=1.0,
            liquidity_floor=0.0,
        )
        manager_double = MarketStateManager(
            n_layers=1,
            n_experts_per_layer=2,
            capital_init=50.0,
            capital_floor=1.0,
            liquidity_floor=0.0,
        )
        manager_single.loss_ema[0] = 2.0
        manager_double.loss_ema[0] = 2.0

        manager_single.settle_layer(
            layer_idx=0,
            winners=torch.tensor([[0]]),
            token_loss=torch.tensor([[1.0]]),
            prices=torch.tensor([0.4, 0.6]),
            shares=torch.tensor([5.0, 5.0]),
            token_weight=torch.ones(1, 1),
            update_state=True,
        )
        manager_double.settle_layer(
            layer_idx=0,
            winners=torch.tensor([[0, 0]]),
            token_loss=torch.tensor([[1.0, 1.0]]),
            prices=torch.tensor([0.4, 0.6]),
            shares=torch.tensor([5.0, 5.0]),
            token_weight=torch.ones(1, 2),
            update_state=True,
        )

        self.assertAlmostEqual(
            float(manager_single.capitals[0, 0].item()),
            float(manager_double.capitals[0, 0].item()),
            places=6,
        )

    def test_reward_critic_masking_only_counts_weighted_winners(self) -> None:
        critic = RewardCritic(dim=4, n_routable=2, hidden_dim=4)
        x = torch.randn(1, 2, 4)
        winners = torch.tensor([[0, 1]])
        realized_reward = torch.tensor([[1.0, 0.0]])
        weight = torch.tensor([[1.0, 0.0]])

        loss = critic.supervised_loss(x, winners, realized_reward, token_weight=weight)
        pred = critic.predict_reward(x)
        chosen = pred.gather(-1, winners.unsqueeze(-1)).squeeze(-1)
        expected = (chosen[:, :1] - realized_reward[:, :1]).pow(2).mean()
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)

    def test_anti_monopoly_price_floor_and_reentry(self) -> None:
        manager = MarketStateManager(
            n_layers=1,
            n_experts_per_layer=2,
            capital_init=100.0,
            capital_floor=1.0,
            price_lr=0.25,
            liquidity_floor=0.10,
        )
        manager.loss_ema[0] = 2.0

        for _ in range(40):
            prices = manager.prices(0)
            manager.settle_layer(
                layer_idx=0,
                winners=torch.tensor([[0, 0, 0, 0]]),
                token_loss=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
                prices=prices,
                shares=(manager.capitals[0] * manager.bet_fraction) / prices.clamp(min=1e-8),
                token_weight=torch.ones(1, 4),
                update_state=True,
            )

        prices = manager.prices(0)
        self.assertLess(float(prices.max().item()), 1.0)
        self.assertGreaterEqual(float(prices.min().item()), 0.10 / 2 - 1e-6)

        router = PredictionMarketRouter(noise_std=0.0)
        pred_reward = torch.tensor([[[float(prices[0].item()), min(float(prices[1].item()) + 0.3, 0.99)]]])
        result = router(
            expert_capitals=manager.capitals[0],
            q=manager.q[0],
            reward_logits=torch.logit(pred_reward.clamp(1e-4, 1 - 1e-4)),
            bet_fraction=manager.bet_fraction,
            price_temperature=manager.price_temperature,
            liquidity_floor=manager.liquidity_floor,
            exploration_epsilon=0.0,
            training=False,
        )
        self.assertEqual(int(result["winners"][0, 0].item()), 1)

    def test_model_integration_smoke(self) -> None:
        torch.manual_seed(0)
        config = CaMoEConfig(
            vocab_size=32,
            dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            n_rosa_experts=1,
            enable_compile=False,
            enable_gradient_checkpointing=False,
            routing_noise_std=0.0,
            exploration_epsilon=0.0,
            uniform_warmup_steps=0,
            seq_len=4,
            total_steps=1,
            batch_size=2,
        )
        model = DummyModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=torch.long)

        result = model(input_ids, targets, training=True, uniform=False)
        self.assertIn("loss", result)
        settle_results = model.settle_all_layers(
            result["loss"].detach(),
            token_weight=result["loss_mask"].float(),
            update_state=True,
        )
        self.assertEqual(len(settle_results), 2)

        optimizer.zero_grad(set_to_none=True)
        critic_loss = model.compute_critic_loss(settle_results, token_weight=result["loss_mask"].float())
        critic_loss.backward()
        optimizer.step()

        metrics = model.market_metrics(valid_mask=result["loss_mask"].float())
        self.assertIn("sequence/layer_0/price_max", metrics)
        self.assertIn("ffn/layer_0/realized_reward_mean", metrics)
        self.assertFalse(torch.isnan(torch.tensor(list(metrics.values()), dtype=torch.float32)).any().item())

    def test_market_weight_zero_matches_uniform_forward(self) -> None:
        torch.manual_seed(0)
        config = CaMoEConfig(
            vocab_size=32,
            dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            n_rosa_experts=1,
            enable_compile=False,
            enable_gradient_checkpointing=False,
            routing_noise_std=0.0,
            exploration_epsilon=0.0,
            uniform_warmup_steps=0,
            seq_len=4,
            total_steps=1,
            batch_size=2,
        )
        model = DummyModel(config)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=torch.long)

        uniform_result = model(input_ids, targets, training=False, uniform=True)
        mixed_result = model(input_ids, targets, training=False, uniform=False, market_weight=0.0)

        self.assertTrue(torch.allclose(uniform_result["logits"], mixed_result["logits"], atol=1e-6, rtol=1e-6))

    def test_shadow_settlement_trains_critic_without_updating_market_state(self) -> None:
        torch.manual_seed(0)
        config = CaMoEConfig(
            vocab_size=32,
            dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            n_rosa_experts=1,
            enable_compile=False,
            enable_gradient_checkpointing=False,
            routing_noise_std=0.0,
            exploration_epsilon=0.0,
            uniform_warmup_steps=10,
            seq_len=4,
            total_steps=1,
            batch_size=2,
        )
        model = DummyModel(config)
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=torch.long)

        warmup_result = model(input_ids, targets, training=True, uniform=True, market_weight=0.0)
        old_ffn_caps = model.ffn_capital_manager.capitals.clone()
        old_ffn_q = model.ffn_capital_manager.q.clone()
        old_seq_caps = model.sequence_capital_manager.capitals.clone()
        old_seq_q = model.sequence_capital_manager.q.clone()

        with torch.no_grad():
            model(input_ids, targets, training=True, uniform=False, market_weight=1.0)
            settle_results = model.settle_all_layers(
                warmup_result["loss"].detach(),
                token_weight=warmup_result["loss_mask"].float(),
                update_state=False,
            )

        critic_loss = model.compute_critic_loss(settle_results, token_weight=warmup_result["loss_mask"].float())
        self.assertGreaterEqual(float(critic_loss.item()), 0.0)
        self.assertTrue(torch.allclose(model.ffn_capital_manager.capitals, old_ffn_caps))
        self.assertTrue(torch.allclose(model.ffn_capital_manager.q, old_ffn_q))
        self.assertTrue(torch.allclose(model.sequence_capital_manager.capitals, old_seq_caps))
        self.assertTrue(torch.allclose(model.sequence_capital_manager.q, old_seq_q))


if __name__ == "__main__":
    unittest.main()
