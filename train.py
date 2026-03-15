"""Prediction-market CaMoE training script."""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Iterator, Literal

import torch
import typer
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from camoe.backbone import init_rwkv7_cuda
from camoe.config import CaMoEConfig, ROSA_BACKEND_CHOICES, get_config
from camoe.model import CaMoE_Model

try:
    import swanlab

    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False


PHASE_IDS = {
    "uniform_warmup": 0.0,
    "full_market": 1.0,
}


def _compute_ste_temperature(step: int, config: CaMoEConfig) -> float:
    if step < config.uniform_warmup_steps:
        return float(config.ste_temperature_start)
    offset = step - config.uniform_warmup_steps
    total = max(config.ste_anneal_steps, 0)
    mid = max(config.ste_midpoint_steps, 0)
    if total == 0:
        return float(config.ste_temperature_end)
    if offset <= mid:
        if mid == 0:
            return float(config.ste_temperature_mid)
        p = offset / mid
        return float(config.ste_temperature_start + (config.ste_temperature_mid - config.ste_temperature_start) * p)
    if offset <= total:
        tail = total - mid
        if tail <= 0:
            return float(config.ste_temperature_end)
        p = (offset - mid) / tail
        return float(config.ste_temperature_mid + (config.ste_temperature_end - config.ste_temperature_mid) * p)
    return float(config.ste_temperature_end)


def get_phase(step: int, config: CaMoEConfig) -> str:
    if step < config.uniform_warmup_steps:
        return "uniform_warmup"
    return "full_market"


def _compute_market_weight(step: int, config: CaMoEConfig) -> float:
    if step < config.uniform_warmup_steps:
        return 0.0
    ramp = max(config.market_ramp_steps, 0)
    if ramp == 0:
        return 1.0
    progress = min(max(step - config.uniform_warmup_steps, 0) / ramp, 1.0)
    return float(progress)


def _compute_exploration_epsilon(step: int, config: CaMoEConfig) -> float:
    if step < config.uniform_warmup_steps:
        return 1.0
    ramp = max(config.market_ramp_steps, 0)
    if ramp == 0:
        return float(config.exploration_epsilon)
    progress = min(max(step - config.uniform_warmup_steps, 0) / ramp, 1.0)
    return float(1.0 - progress * (1.0 - config.exploration_epsilon))


def _set_exploration_epsilon(model: CaMoE_Model, exploration_epsilon: float) -> None:
    if model.sequence_capital_manager is not None:
        model.sequence_capital_manager.exploration_epsilon = float(exploration_epsilon)
    model.ffn_capital_manager.exploration_epsilon = float(exploration_epsilon)


def load_training_split(path: str) -> Dataset:
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            first_key = next(iter(dataset.keys()))
            dataset = dataset[first_key]
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")
    dataset.set_format(type="torch", columns=["input_ids"])
    return dataset


def build_collate_fn(seq_len: int, ignore_index: int):
    chunk_len = 16

    def collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        max_len = min(max(len(ids) for ids in input_ids), seq_len + 1)
        input_len = ((max_len - 1 + chunk_len - 1) // chunk_len) * chunk_len
        target_len = max(input_len + 1, chunk_len + 1)
        padded = torch.full((len(batch), target_len), ignore_index, dtype=torch.long)
        for row, ids in enumerate(input_ids):
            length = min(len(ids), target_len)
            padded[row, :length] = ids[:length]
        return padded

    return collate


def infinite_loader(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


def split_optim_params(model: CaMoE_Model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    expert_params = []
    critic_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "reward_critic" in name:
            critic_params.append(param)
        else:
            expert_params.append(param)
    return expert_params, critic_params


def save_checkpoint(
    path: str,
    model: CaMoE_Model,
    expert_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    step: int,
    swanlab_run_id: str | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "expert_optimizer": expert_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "step": step,
        "config": model.get_checkpoint_config(),
    }
    if swanlab_run_id is not None:
        payload["swanlab_run_id"] = swanlab_run_id
    torch.save(payload, path)


def main(
    scale: Literal["0.1b", "0.4b"] = typer.Option("0.4b"),
    data: str = typer.Option(..., help="Path to a HuggingFace dataset saved with load_from_disk"),
    save_dir: str = typer.Option("checkpoints/v23"),
    resume: str | None = typer.Option(None),
    device: str = typer.Option("cuda"),
    batch_size: int | None = typer.Option(None),
    seq_len: int | None = typer.Option(None),
    steps: int | None = typer.Option(None),
    lr: float | None = typer.Option(None),
    critic_lr: float | None = typer.Option(None),
    critic_update_interval: int | None = typer.Option(None),
    bet_fraction: float | None = typer.Option(None),
    price_lr: float | None = typer.Option(None),
    price_temperature: float | None = typer.Option(None),
    liquidity_floor: float | None = typer.Option(None),
    reward_scale: float | None = typer.Option(None),
    reward_eps: float | None = typer.Option(None),
    reward_hidden_dim: int | None = typer.Option(None),
    uniform_warmup_steps: int | None = typer.Option(None),
    market_ramp_steps: int | None = typer.Option(None),
    routing_noise_std: float | None = typer.Option(None),
    exploration_epsilon: float | None = typer.Option(None),
    n_deepembed_experts: int | None = typer.Option(None),
    n_slim_deepembed_experts: int | None = typer.Option(None),
    deepembed_mode: Literal["1x", "4x"] | None = typer.Option(None),
    deepembed_expand: int | None = typer.Option(None),
    slim_deepembed_rank: int | None = typer.Option(None),
    n_rosa_experts: int | None = typer.Option(None),
    rosa_backend: str | None = typer.Option(
        None,
        help=f"Canonical backends: {', '.join(ROSA_BACKEND_CHOICES)}. Old aliases still work.",
    ),
    rosa_hard_backend: str | None = typer.Option(
        None,
        help=f"Canonical backends: {', '.join(ROSA_BACKEND_CHOICES)}. Old aliases still work.",
    ),
    rosa_hard_switch_step: int | None = typer.Option(None),
    rosa_bits: int | None = typer.Option(None),
    slim_rosa_heads: int | None = typer.Option(None),
    rosa_truncation_length: int | None = typer.Option(None),
    rosa_native_mode: bool | None = typer.Option(None, "--rosa-native-mode/--no-rosa-native-mode"),
    rosa_use_gate: bool | None = typer.Option(None, "--rosa-use-gate/--no-rosa-use-gate"),
    market_alpha_start: float | None = typer.Option(None),
    market_alpha_end: float | None = typer.Option(None),
    routing_entropy_reg: float | None = typer.Option(None),
    critic_shadow_prewarm: bool | None = typer.Option(None, "--critic-shadow-prewarm/--no-critic-shadow-prewarm"),
    critic_shadow_market: bool | None = typer.Option(None, "--critic-shadow-market/--no-critic-shadow-market"),
    routing_ste: bool | None = typer.Option(None, "--routing-ste/--no-routing-ste"),
    ste_temperature_start: float | None = typer.Option(None),
    ste_temperature_mid: float | None = typer.Option(None),
    ste_temperature_end: float | None = typer.Option(None),
    ste_midpoint_steps: int | None = typer.Option(None),
    ste_anneal_steps: int | None = typer.Option(None),
    compile_enabled: bool | None = typer.Option(None, "--compile/--no-compile"),
    gradient_checkpointing: bool | None = typer.Option(
        None,
        "--gradient-checkpointing/--no-gradient-checkpointing",
    ),
    log_interval: int = typer.Option(100),
    save_interval: int = typer.Option(1000),
    num_workers: int = typer.Option(0),
    amp: bool = typer.Option(False, "--amp/--no-amp"),
) -> None:
    args = SimpleNamespace(
        scale=scale,
        data=data,
        save_dir=save_dir,
        resume=resume,
        device=device,
        log_interval=log_interval,
        save_interval=save_interval,
        num_workers=num_workers,
        amp=amp,
    )

    config = get_config(scale).with_overrides(
        batch_size=batch_size,
        seq_len=seq_len,
        total_steps=steps,
        lr=lr,
        critic_lr=critic_lr,
        critic_update_interval=critic_update_interval,
        bet_fraction=bet_fraction,
        price_lr=price_lr,
        price_temperature=price_temperature,
        liquidity_floor=liquidity_floor,
        reward_scale=reward_scale,
        reward_eps=reward_eps,
        reward_hidden_dim=reward_hidden_dim,
        uniform_warmup_steps=uniform_warmup_steps,
        market_ramp_steps=market_ramp_steps,
        routing_noise_std=routing_noise_std,
        exploration_epsilon=exploration_epsilon,
        n_deepembed_experts=n_deepembed_experts,
        n_slim_deepembed_experts=n_slim_deepembed_experts,
        deepembed_mode=deepembed_mode,
        deepembed_expand=deepembed_expand,
        slim_deepembed_rank=slim_deepembed_rank,
        n_rosa_experts=n_rosa_experts,
        rosa_backend=rosa_backend,
        rosa_hard_backend=rosa_hard_backend,
        rosa_hard_switch_step=rosa_hard_switch_step,
        rosa_bits=rosa_bits,
        slim_rosa_heads=slim_rosa_heads,
        rosa_truncation_length=rosa_truncation_length,
        rosa_native_mode=rosa_native_mode,
        rosa_use_gate=rosa_use_gate,
        market_alpha_start=market_alpha_start,
        market_alpha_end=market_alpha_end,
        routing_entropy_reg=routing_entropy_reg,
        critic_shadow_prewarm=critic_shadow_prewarm,
        critic_shadow_market=critic_shadow_market,
        routing_ste=routing_ste,
        ste_temperature_start=ste_temperature_start,
        ste_temperature_mid=ste_temperature_mid,
        ste_temperature_end=ste_temperature_end,
        ste_midpoint_steps=ste_midpoint_steps,
        ste_anneal_steps=ste_anneal_steps,
        enable_compile=compile_enabled,
        enable_gradient_checkpointing=gradient_checkpointing,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_ctx = (
        lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp_enabled
        else nullcontext
    )

    init_rwkv7_cuda()

    model = CaMoE_Model(config).to(device)
    expert_params, critic_params = split_optim_params(model)
    expert_optimizer = torch.optim.AdamW(expert_params, lr=config.lr)
    critic_optimizer = torch.optim.Adam(critic_params, lr=config.critic_lr)

    start_step = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "expert_optimizer" in checkpoint:
            expert_optimizer.load_state_dict(checkpoint["expert_optimizer"])
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_step = int(checkpoint.get("step", 0)) + 1

    swanlab_run_id = None
    if HAS_SWANLAB:
        resume_id = checkpoint.get("swanlab_run_id") if isinstance(checkpoint, dict) else None
        experiment = swanlab.init(
            project="CaMoE-v23",
            name=f"prediction-market-{args.scale}",
            config=config.to_dict(),
            id=resume_id,
            resume="allow",
        )
        swanlab_run_id = experiment.public.run_id

    dataset = load_training_split(args.data)
    collate_fn = build_collate_fn(config.seq_len, config.ignore_index)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    train_iter = infinite_loader(loader)

    model.train()
    for step in range(start_step, config.total_steps):
        step_start = time.time()
        batch = next(train_iter).to(device)
        if batch.size(1) <= 1:
            continue

        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        phase = get_phase(step, config)
        uniform = phase == "uniform_warmup"
        ste_temperature = _compute_ste_temperature(step, config)
        dynamic_eps = _compute_exploration_epsilon(step, config)
        market_weight = _compute_market_weight(step, config)
        _set_exploration_epsilon(model, dynamic_eps)

        expert_optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            result = model(
                input_ids,
                targets,
                ste_temperature=ste_temperature,
                training=True,
                uniform=uniform,
                market_weight=market_weight,
                current_step=step,
            )
        result["loss_scalar"].backward()
        clip_grad_norm_(expert_params, config.grad_clip)
        expert_optimizer.step()

        with torch.no_grad():
            settle_results = model.settle_all_layers(
                result["loss"].detach(),
                token_weight=result["loss_mask"].detach().float(),
                update_state=not uniform,
            )

        critic_loss_value = None
        if settle_results and not uniform:
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = model.compute_critic_loss(
                settle_results,
                token_weight=result["loss_mask"].detach().float(),
            )
            critic_loss.backward()
            clip_grad_norm_(critic_params, 1.0)
            critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach().item())

        if step % args.log_interval == 0:
            metrics = model.market_metrics(valid_mask=result["loss_mask"].detach().float())
            entropy_values = [value for key, value in metrics.items() if key.endswith("routing_entropy")]
            routing_entropy = sum(entropy_values) / max(len(entropy_values), 1) if entropy_values else 0.0
            tps = input_ids.numel() / max(time.time() - step_start, 1e-6)
            print(
                f"step={step} phase={phase} loss={float(result['loss_scalar'].detach().item()):.4f} "
                f"critic_loss={critic_loss_value if critic_loss_value is not None else 'n/a'} "
                f"routing_entropy={routing_entropy:.4f}"
            )
            if HAS_SWANLAB:
                logs = {
                    "Loss/Train_Main": float(result.get("loss_main_scalar", result["loss_scalar"]).detach().item()),
                    "Loss/Train_Total": float(result["loss_scalar"].detach().item()),
                    "Loss/Train_Critic": float(critic_loss_value or 0.0),
                    "Loss/Aux_Balance": 0.0,
                    "Speed/TPS": float(tps),
                    "Phase/ID": PHASE_IDS.get(phase, -1.0),
                    f"Phase/{phase}": 1.0,
                    "Market/ExplorationEpsilon": float(dynamic_eps),
                    "Market/RoutingEntropy": float(routing_entropy),
                    "Market/Weight": float(market_weight),
                    "Market/STETemperature": float(ste_temperature),
                    "ROSA/HardSwitchStep": float(config.rosa_hard_switch_step or -1),
                    "Runtime/CompileEnabled": float(config.enable_compile),
                    "Runtime/GradientCheckpointing": float(config.enable_gradient_checkpointing),
                }
                if config.n_rosa_experts > 0 and model.blocks and model.blocks[0].rosa_expert is not None:
                    logs["ROSA/UsingHardBackend"] = float(
                        model.blocks[0].rosa_expert.effective_backend(step) == (config.rosa_hard_backend or "")
                    )
                logs.update(metrics)
                swanlab.log(logs, step=step)

        if step > 0 and step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"v23_step{step}.pth")
            save_checkpoint(
                checkpoint_path,
                model,
                expert_optimizer,
                critic_optimizer,
                step,
                swanlab_run_id=swanlab_run_id,
            )

    final_path = os.path.join(args.save_dir, "v23_final.pth")
    save_checkpoint(
        final_path,
        model,
        expert_optimizer,
        critic_optimizer,
        config.total_steps - 1,
        swanlab_run_id=swanlab_run_id,
    )


if __name__ == "__main__":
    typer.run(main)
