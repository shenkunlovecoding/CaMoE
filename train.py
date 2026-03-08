"""CaMoE v22 training script."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from typing import Iterator

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from camoe.backbone import init_rwkv7_cuda
from camoe.config import CONFIG_0_1B, CONFIG_0_4B, CaMoEConfig
from camoe.model import CaMoE_Model


def get_phase(step: int, config: CaMoEConfig) -> tuple[str, float]:
    s1 = config.prewarm_steps
    s2 = s1 + config.market_warmup_steps
    s3 = s2 + config.critic_warmup_steps
    if step < s1:
        return "prewarm", 0.0
    if step < s2:
        return "market_warm", 0.0
    if step < s3:
        alpha = (step - s2) / max(config.critic_warmup_steps, 1)
        return "critic_warm", float(alpha)
    return "full_market", 1.0


def select_config(scale: str) -> CaMoEConfig:
    if scale == "0.1b":
        return CONFIG_0_1B.copy()
    if scale == "0.4b":
        return CONFIG_0_4B.copy()
    raise ValueError(f"Unknown scale: {scale}")


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
        if "critic_pair" in name:
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
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "expert_optimizer": expert_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "step": step,
            "config": model.get_checkpoint_config(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CaMoE v22 pure-market model")
    parser.add_argument("--scale", default="0.4b", choices=["0.1b", "0.4b"])
    parser.add_argument("--data", required=True, help="Path to a HuggingFace dataset saved with load_from_disk")
    parser.add_argument("--save_dir", default="checkpoints/v22")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--critic_update_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    config = select_config(args.scale)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.seq_len is not None:
        config.seq_len = args.seq_len
    if args.steps is not None:
        config.total_steps = args.steps
    if args.lr is not None:
        config.lr = args.lr
    if args.critic_lr is not None:
        config.critic_lr = args.critic_lr
    if args.critic_update_interval is not None:
        config.critic_update_interval = args.critic_update_interval

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
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "expert_optimizer" in checkpoint:
            expert_optimizer.load_state_dict(checkpoint["expert_optimizer"])
        if "critic_optimizer" in checkpoint:
            critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        start_step = int(checkpoint.get("step", 0)) + 1

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
        batch = next(train_iter).to(device)
        if batch.size(1) <= 1:
            continue

        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        phase, critic_alpha = get_phase(step, config)

        expert_optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            result = model(
                input_ids,
                targets,
                critic_alpha=critic_alpha,
                training=True,
                uniform=(phase == "prewarm"),
            )
        result["loss_scalar"].backward()
        clip_grad_norm_(expert_params, config.grad_clip)
        expert_optimizer.step()

        settle_results = []
        if phase != "prewarm":
            with torch.no_grad():
                settle_results = model.settle_all_layers(result["loss"].detach())

        critic_loss_value = None
        if phase not in ("prewarm", "market_warm") and settle_results and step % config.critic_update_interval == 0:
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = model.compute_critic_loss(settle_results)
            critic_loss.backward()
            clip_grad_norm_(critic_params, 1.0)
            critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach().item())

        if step % args.log_interval == 0:
            metrics = model.market_metrics()
            entropy_values = [value for key, value in metrics.items() if key.endswith("routing_entropy")]
            routing_entropy = sum(entropy_values) / max(len(entropy_values), 1) if entropy_values else 0.0
            print(
                f"step={step} phase={phase} loss={float(result['loss_scalar'].detach().item()):.4f} "
                f"critic_loss={critic_loss_value if critic_loss_value is not None else 'n/a'} "
                f"routing_entropy={routing_entropy:.4f}"
            )

        if step > 0 and step % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f"v22_step{step}.pth")
            save_checkpoint(checkpoint_path, model, expert_optimizer, critic_optimizer, step)

    final_path = os.path.join(args.save_dir, "v22_final.pth")
    save_checkpoint(final_path, model, expert_optimizer, critic_optimizer, config.total_steps - 1)


if __name__ == "__main__":
    main()
