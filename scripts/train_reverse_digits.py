"""Train tiny toy-sequence baselines for ROSA and CaMoE."""

from __future__ import annotations

import argparse
import html
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camoe.config import CaMoEConfig
from camoe.model import CaMoE_Model
from camoe.reverse_baselines import SingleROSAReverseModel, SingleRWKVReverseModel

try:
    import swanlab

    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

PAD_ID = 0
BOS_ID = 1
SEP_ID = 2
EOS_ID = 3
REV_ID = 14
COPY_ID = 15
PARITY_ID = 16
CUMSUM_ID = 17
MAJORITY_ID = 18
COUNT_ID = 19
IGNORE_INDEX = -100
TOKEN_NAMES = {
    PAD_ID: "[PAD]",
    BOS_ID: "[BOS]",
    SEP_ID: "[SEP]",
    EOS_ID: "[EOS]",
    REV_ID: "[REV]",
    COPY_ID: "[COPY]",
    PARITY_ID: "[PARITY]",
    CUMSUM_ID: "[CUMSUM]",
    MAJORITY_ID: "[MAJORITY]",
    COUNT_ID: "[COUNT]",
}
TOKEN_NAMES.update({4 + value: str(value) for value in range(10)})
ANSI_BLUE = "\033[94m"
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
OPERATION_NAME_BY_ID = {
    0: "reverse_digits",
    1: "copy_digits",
    2: "parity_digits",
    3: "cumsum_mod10",
    4: "majority_vote",
    5: "count_ones",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_task_name(path: str, train_dataset: Dataset) -> str:
    if len(train_dataset) > 0:
        row = train_dataset[0]
        task_name = row.get("task")
        if isinstance(task_name, str) and task_name:
            return task_name
    return Path(path).name


def load_splits(path: str) -> tuple[str, Dataset, Dataset, Dataset]:
    dataset = load_from_disk(path)
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {path}, got {type(dataset)!r}")
    train_dataset = dataset["train"]
    return infer_task_name(path, train_dataset), train_dataset, dataset["validation"], dataset["ood"]


def get_viz_samples(dataset: Dataset, task_name: str) -> list[dict]:
    if not task_name.startswith("mixed"):
        return [dataset[idx] for idx in range(min(4, len(dataset)))]

    op_order = [name for name in OPERATION_NAME_BY_ID.values()]
    present_ops: list[str] = []
    for row in dataset:
        op_name = row.get("operation") or row.get("op") or row.get("task")
        if op_name in op_order and op_name not in present_ops:
            present_ops.append(op_name)

    max_samples = min(max(4, len(present_ops)), len(dataset))
    if not present_ops:
        return [dataset[idx] for idx in range(max_samples)]

    per_op = max(1, max_samples // len(present_ops))
    selected_by_op = {op_name: [] for op_name in present_ops}
    seen_inputs: set[tuple[int, ...]] = set()

    for row in dataset:
        op_name = row.get("operation") or row.get("op") or row.get("task")
        if op_name not in selected_by_op or len(selected_by_op[op_name]) >= per_op:
            continue
        selected_by_op[op_name].append(row)
        seen_inputs.add(tuple(int(token) for token in row["input_ids"]))
        if sum(len(rows) for rows in selected_by_op.values()) >= per_op * len(present_ops):
            break

    samples: list[dict] = []
    for op_name in op_order:
        samples.extend(selected_by_op.get(op_name, []))

    if len(samples) < max_samples:
        for row in dataset:
            key = tuple(int(token) for token in row["input_ids"])
            if key in seen_inputs:
                continue
            samples.append(row)
            seen_inputs.add(key)
            if len(samples) >= max_samples:
                break
    return samples[:max_samples]


def pad_to_chunk(length: int, chunk_len: int = 16) -> int:
    if length % chunk_len == 0:
        return length
    return ((length + chunk_len - 1) // chunk_len) * chunk_len


def build_collate_fn() -> callable:
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        padded_len = pad_to_chunk(max_len)
        batch_size = len(batch)
        input_ids = torch.full((batch_size, padded_len), PAD_ID, dtype=torch.long)
        targets = torch.full((batch_size, padded_len), IGNORE_INDEX, dtype=torch.long)
        supervised_mask = torch.zeros((batch_size, padded_len), dtype=torch.float32)
        valid_mask = torch.zeros((batch_size, padded_len), dtype=torch.float32)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        operation_id = torch.full((batch_size,), -1, dtype=torch.long)

        for row_idx, row in enumerate(batch):
            length = len(row["input_ids"])
            input_ids[row_idx, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            targets[row_idx, :length] = torch.tensor(row["targets"], dtype=torch.long)
            supervised_mask[row_idx, :length] = torch.tensor(row["supervised_mask"], dtype=torch.float32)
            valid_mask[row_idx, :length] = 1.0
            lengths[row_idx] = int(row["length"])
            operation_id[row_idx] = int(row.get("operation_id", -1))

        return {
            "input_ids": input_ids,
            "targets": targets,
            "supervised_mask": supervised_mask,
            "valid_mask": valid_mask,
            "length": lengths,
            "operation_id": operation_id,
        }

    return collate


def infinite_loader(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def masked_metrics(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.argmax(dim=-1)
    valid = valid_mask > 0
    correct = (predictions == targets) & valid
    token_acc = correct.sum().float() / valid.sum().clamp(min=1).float()
    exact_match = correct.logical_or(~valid).all(dim=1).float().mean()
    return token_acc, exact_match


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


def build_config(args: argparse.Namespace) -> CaMoEConfig:
    return CaMoEConfig(
        vocab_size=20,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_experts=2,
        n_deepembed_experts=args.n_deepembed_experts,
        n_slim_deepembed_experts=args.n_slim_deepembed_experts,
        n_rosa_experts=1,
        ffn_expand=4,
        tie_weights=True,
        rosa_backend="wind",
        rosa_bits=args.rosa_bits,
        slim_rosa_heads=args.slim_rosa_heads,
        rosa_truncation_length=args.rosa_truncation_length,
        deepembed_mode=args.deepembed_mode,
        deepembed_expand=args.deepembed_expand,
        slim_deepembed_rank=args.slim_deepembed_rank,
        auction_noise_std=args.auction_noise_std,
        enable_compile=False,
        enable_gradient_checkpointing=False,
        expert_capital_init=1.0,
        critic_capital_init=0.5,
        ema_decay=0.95,
        capital_floor=0.25,
        capital_ceiling=args.capital_ceiling,
        depreciation=args.depreciation,
        critic_hidden_dim=None,
        critic_update_interval=args.critic_update_interval,
        critic_lr=args.critic_lr,
        prewarm_steps=args.prewarm_steps,
        market_warmup_steps=args.market_warmup_steps,
        critic_warmup_steps=args.critic_warmup_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=pad_to_chunk(2 * args.max_eval_length + 3),
        total_steps=args.steps,
        grad_clip=1.0,
        ignore_index=IGNORE_INDEX,
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_kind: str,
    critic_alpha: float = 1.0,
    uniform: bool = False,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_correct = 0.0
    token_total = 0.0
    exact_total = 0.0
    sample_total = 0.0
    token_correct_by_op = {name: 0.0 for name in OPERATION_NAME_BY_ID.values()}
    token_total_by_op = {name: 0.0 for name in OPERATION_NAME_BY_ID.values()}
    exact_total_by_op = {name: 0.0 for name in OPERATION_NAME_BY_ID.values()}
    sample_total_by_op = {name: 0.0 for name in OPERATION_NAME_BY_ID.values()}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if model_kind == "camoe":
                result = model(
                    batch["input_ids"],
                    batch["targets"],
                    critic_alpha=critic_alpha,
                    training=False,
                    uniform=uniform,
                )
            else:
                result = model(batch["input_ids"], batch["targets"])

            valid_mask = batch["supervised_mask"]
            loss_sum += float((result["loss"] * valid_mask).sum().item())
            token_acc, exact = masked_metrics(result["logits"], batch["targets"], valid_mask)
            token_total += float(valid_mask.sum().item())
            token_correct += float(token_acc.item() * max(valid_mask.sum().item(), 1.0))
            exact_total += float(exact.item() * batch["input_ids"].size(0))
            sample_total += float(batch["input_ids"].size(0))
            op_ids = batch["operation_id"]
            if (op_ids >= 0).any():
                predictions = result["logits"].argmax(dim=-1)
                valid = valid_mask > 0
                correct = (predictions == batch["targets"]) & valid
                for op_id, op_name in OPERATION_NAME_BY_ID.items():
                    sample_mask = op_ids == op_id
                    if not sample_mask.any():
                        continue
                    op_valid = valid[sample_mask]
                    op_correct = correct[sample_mask]
                    token_total_by_op[op_name] += float(op_valid.sum().item())
                    token_correct_by_op[op_name] += float(op_correct.sum().item())
                    op_exact = op_correct.logical_or(~op_valid).all(dim=1).float()
                    exact_total_by_op[op_name] += float(op_exact.sum().item())
                    sample_total_by_op[op_name] += float(sample_mask.sum().item())

    model.train()
    denom = max(token_total, 1.0)
    metrics = {
        "loss": loss_sum / denom,
        "answer_token_acc": token_correct / denom,
        "exact_match": exact_total / max(sample_total, 1.0),
    }
    for op_name in OPERATION_NAME_BY_ID.values():
        if sample_total_by_op[op_name] <= 0:
            continue
        metrics[f"{op_name}/answer_token_acc"] = token_correct_by_op[op_name] / max(token_total_by_op[op_name], 1.0)
        metrics[f"{op_name}/exact_match"] = exact_total_by_op[op_name] / sample_total_by_op[op_name]
    return metrics


def decode_display_stream(input_ids: torch.Tensor, targets: torch.Tensor) -> list[str]:
    display_ids: list[int] = []
    input_list = input_ids.tolist()
    target_list = targets.tolist()
    for idx, token_id in enumerate(target_list):
        if token_id != IGNORE_INDEX:
            display_ids.append(int(token_id))
        elif idx + 1 < len(input_list):
            display_ids.append(int(input_list[idx + 1]))
        else:
            display_ids.append(IGNORE_INDEX)
    tokens = []
    for token_id in display_ids:
        if token_id == IGNORE_INDEX:
            tokens.append("·")
        else:
            tokens.append(TOKEN_NAMES.get(int(token_id), f"<{int(token_id)}>"))
    return tokens


def colorize_token(token: str, winner: int, is_answer: bool) -> str:
    if token == "[SEP]":
        return f"{ANSI_YELLOW}{token}{ANSI_RESET}"
    color = ANSI_RED if winner == 1 else ANSI_BLUE
    decorated = f"{token}" if is_answer else f"({token})"
    return f"{color}{decorated}{ANSI_RESET}"


def render_route_preview(
    model: CaMoE_Model,
    batch: dict[str, torch.Tensor],
    artifact_dir: Path,
    step: int,
    task_name: str,
    critic_alpha: float,
    uniform: bool,
) -> None:
    model.eval()
    with torch.no_grad():
        model(
            batch["input_ids"],
            batch["targets"],
            critic_alpha=critic_alpha,
            training=False,
            uniform=uniform,
        )
    diagnostics = model.get_market_diagnostics()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    html_path = artifact_dir / f"step_{step:06d}.html"

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Consolas,monospace;background:#111;color:#eee;padding:16px;}",
        ".rwkv{color:#62b0ff;} .rosa{color:#ff6b6b;} .sep{color:#ffd166;font-weight:700;}",
        ".answer{font-weight:700;} .prefix{opacity:0.65;} table{border-collapse:collapse;margin:12px 0;}",
        "td,th{border:1px solid #333;padding:4px 6px;} h2,h3{margin:16px 0 8px;}",
        "</style></head><body>",
        f"<h1>{html.escape(task_name)} Routing Preview @ step {step}</h1>",
    ]

    targets_cpu = batch["targets"].cpu()
    input_ids_cpu = batch["input_ids"].cpu()
    answer_mask_cpu = batch["supervised_mask"].cpu()
    valid_mask_cpu = batch["valid_mask"].cpu()
    op_ids_cpu = batch["operation_id"].cpu()

    for entry in diagnostics:
        layer_idx = int(entry["layer"])
        market_name = str(entry.get("market", "ffn"))
        cache = entry["cache"]
        winners = cache["winners"].cpu()
        bids = cache.get("bids")
        bids = bids.cpu() if bids is not None else None
        html_parts.append(f"<h2>Layer {layer_idx} | {html.escape(market_name)}</h2>")
        print(f"[route-preview] task={task_name} layer={layer_idx} market={market_name}")
        for sample_idx in range(targets_cpu.size(0)):
            tokens = decode_display_stream(input_ids_cpu[sample_idx], targets_cpu[sample_idx])
            winners_row = winners[sample_idx].tolist()
            answer_row = answer_mask_cpu[sample_idx].tolist()
            valid_row = valid_mask_cpu[sample_idx].tolist()
            if bids is None:
                selected_bids = None
            else:
                selected_bids = bids[sample_idx, torch.arange(winners.size(1)), winners[sample_idx]].tolist()
            op_name = OPERATION_NAME_BY_ID.get(int(op_ids_cpu[sample_idx].item()), "unknown")

            ansi_tokens = []
            html_tokens = []
            rows = []
            bid_iter = selected_bids if selected_bids is not None else [None] * len(tokens)
            for pos, (token, winner, is_answer, is_valid, bid_value) in enumerate(
                zip(tokens, winners_row, answer_row, valid_row, bid_iter)
            ):
                if not is_valid:
                    continue
                ansi_tokens.append(colorize_token(token, int(winner), bool(is_answer)))
                classes = ["sep" if token == "[SEP]" else ("rosa" if int(winner) == 1 else "rwkv")]
                classes.append("answer" if is_answer else "prefix")
                html_tokens.append(f"<span class=\"{' '.join(classes)}\">{html.escape(token)}</span>")
                bid_display = "N/A" if bid_value is None else f"{float(bid_value):.4f}"
                rows.append(
                    "<tr>"
                    f"<td>{pos}</td><td>{html.escape(token)}</td><td>{int(winner)}</td>"
                    f"<td>{bid_display}</td><td>{int(bool(is_answer))}</td>"
                    "</tr>"
                )

            print(f"  sample={sample_idx} op={op_name} {' '.join(ansi_tokens)}")
            html_parts.append(f"<h3>Sample {sample_idx} | {html.escape(op_name)}</h3>")
            html_parts.append(f"<div>{' '.join(html_tokens)}</div>")
            html_parts.append("<table><tr><th>pos</th><th>token</th><th>winner</th><th>bid</th><th>answer</th></tr>")
            html_parts.extend(rows)
            html_parts.append("</table>")

    html_parts.append("</body></html>")
    html_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"[route-preview] saved={html_path}")
    model.train()


def preview_route_stats(model: CaMoE_Model, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    diagnostics = model.get_market_diagnostics()
    if not diagnostics:
        return {}

    stats: dict[str, float] = {}
    op_ids = batch["operation_id"]
    answer_mask = batch["supervised_mask"]
    valid_mask = batch["valid_mask"]

    for entry in diagnostics:
        layer_idx = int(entry["layer"])
        cache = entry["cache"]
        winners = cache["winners"]
        expert_types = entry["expert_types"]
        rosa_indices = [idx for idx, kind in enumerate(expert_types) if kind == "rosa"]
        if not rosa_indices:
            continue

        for op_id, op_name in OPERATION_NAME_BY_ID.items():
            sample_mask = (op_ids == op_id).float().unsqueeze(1)
            if sample_mask.sum().item() <= 0:
                continue
            token_mask = answer_mask * valid_mask * sample_mask
            if token_mask.sum().item() <= 0:
                continue
            winner_one_hot = torch.nn.functional.one_hot(winners, num_classes=len(expert_types)).to(dtype=torch.float32)
            rosa_share = winner_one_hot[:, :, rosa_indices].sum(dim=-1)
            rosa_share = (rosa_share * token_mask).sum() / token_mask.sum().clamp(min=1e-8)
            stats[f"preview/{op_name}/layer_{layer_idx}/rosa_share_answer"] = float(rosa_share.item())
    return stats


def market_logs_from_batch(
    model: CaMoE_Model,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    metrics = model.market_metrics(token_mask=batch["supervised_mask"], valid_mask=batch["valid_mask"])
    logs: dict[str, float] = {}
    for layer_idx, block in enumerate(model.blocks):
        expert_types = [expert.expert_type for expert in block.experts]
        for expert_idx, expert_type in enumerate(expert_types):
            logs[f"market/layer_{layer_idx}/capital_{expert_type}"] = metrics.get(
                f"layer_{layer_idx}/expert_{expert_idx}/capital",
                0.0,
            )
            logs[f"market/layer_{layer_idx}/winner_share_{expert_type}_answer"] = metrics.get(
                f"layer_{layer_idx}/expert_{expert_idx}/winner_share_answer",
                0.0,
            )
            logs[f"market/layer_{layer_idx}/bid_{expert_type}_prefix"] = metrics.get(
                f"layer_{layer_idx}/expert_{expert_idx}/bid_mean_prefix",
                0.0,
            )
            logs[f"market/layer_{layer_idx}/bid_{expert_type}_answer"] = metrics.get(
                f"layer_{layer_idx}/expert_{expert_idx}/bid_mean_answer",
                0.0,
            )
            logs[f"market/layer_{layer_idx}/critic_pos_{expert_type}_answer"] = metrics.get(
                f"layer_{layer_idx}/expert_{expert_idx}/position_mean_answer",
                0.0,
            )
        logs[f"market/layer_{layer_idx}/capital_gini"] = metrics.get(f"layer_{layer_idx}/capital_gini", 0.0)
        logs[f"market/layer_{layer_idx}/routing_entropy_all"] = metrics.get(
            f"layer_{layer_idx}/routing_entropy_all",
            0.0,
        )
        logs[f"market/layer_{layer_idx}/routing_entropy_answer"] = metrics.get(
            f"layer_{layer_idx}/routing_entropy_answer",
            0.0,
        )
    op_ids = batch.get("operation_id")
    if op_ids is not None and (op_ids >= 0).any():
        for op_id, op_name in OPERATION_NAME_BY_ID.items():
            sample_mask = (op_ids == op_id).float().unsqueeze(1)
            if sample_mask.sum().item() == 0:
                continue
            op_metrics = model.market_metrics(
                token_mask=batch["supervised_mask"] * sample_mask,
                valid_mask=batch["valid_mask"] * sample_mask,
            )
            for layer_idx, block in enumerate(model.blocks):
                expert_types = [expert.expert_type for expert in block.experts]
                logs[f"market/{op_name}/layer_{layer_idx}/routing_entropy_answer"] = op_metrics.get(
                    f"layer_{layer_idx}/routing_entropy_answer",
                    0.0,
                )
                for expert_idx, expert_type in enumerate(expert_types):
                    logs[f"market/{op_name}/layer_{layer_idx}/winner_share_{expert_type}_answer"] = op_metrics.get(
                        f"layer_{layer_idx}/expert_{expert_idx}/winner_share_answer",
                        0.0,
                    )
    return logs


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: CaMoEConfig,
    critic_optimizer: torch.optim.Optimizer | None = None,
    swanlab_run_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
    }
    if critic_optimizer is not None:
        payload["critic_optimizer"] = critic_optimizer.state_dict()
    if swanlab_run_id is not None:
        payload["swanlab_run_id"] = swanlab_run_id
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train small toy-sequence ROSA/CaMoE models")
    parser.add_argument("--data_dir", type=str, default="data/reverse_digits")
    parser.add_argument("--model_kind", choices=["single_rosa", "single_rwkv", "camoe"], default="camoe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--critic_update_interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_deepembed_experts", type=int, default=0)
    parser.add_argument("--n_slim_deepembed_experts", type=int, default=0)
    parser.add_argument("--deepembed_mode", type=str, default="1x", choices=["1x", "4x"])
    parser.add_argument("--deepembed_expand", type=int, default=4)
    parser.add_argument("--slim_deepembed_rank", type=int, default=32)
    parser.add_argument("--slim_rosa_heads", type=int, default=8)
    parser.add_argument("--rosa_bits", type=int, default=8)
    parser.add_argument("--rosa_truncation_length", type=int, default=8)
    parser.add_argument("--auction_noise_std", type=float, default=0.05)
    parser.add_argument("--depreciation", type=float, default=1e-3)
    parser.add_argument("--capital_ceiling", type=float, default=10.0)
    parser.add_argument("--prewarm_steps", type=int, default=100)
    parser.add_argument("--market_warmup_steps", type=int, default=100)
    parser.add_argument("--critic_warmup_steps", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="checkpoints/reverse_digits")
    parser.add_argument("--artifact_dir", type=str, default="artifacts/reverse_digits")
    parser.add_argument("--max_eval_length", type=int, default=20)
    parser.add_argument("--no_swanlab", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    task_name, train_ds, val_ds, ood_ds = load_splits(args.data_dir)
    collate_fn = build_collate_fn()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    ood_loader = DataLoader(
        ood_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    preview_rows = get_viz_samples(val_ds, task_name)
    preview_batch = move_batch(collate_fn(preview_rows), device)

    config = build_config(args)
    if args.model_kind == "camoe":
        model: torch.nn.Module = CaMoE_Model(config).to(device)
        expert_params, critic_params = split_optim_params(model)
        optimizer = torch.optim.AdamW(expert_params, lr=args.lr, weight_decay=args.weight_decay)
        critic_optimizer = torch.optim.AdamW(critic_params, lr=args.critic_lr, weight_decay=args.weight_decay)
    elif args.model_kind == "single_rwkv":
        model = SingleRWKVReverseModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        critic_optimizer = None
    else:
        model = SingleROSAReverseModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        critic_optimizer = None

    swanlab_run_id = None
    if HAS_SWANLAB and not args.no_swanlab:
        experiment = swanlab.init(
            project="CaMoE-toy-seq",
            name=f"{args.model_kind}-{task_name}",
            config={
                "task_name": task_name,
                "model_kind": args.model_kind,
                "config": config.to_dict(),
                "data_dir": args.data_dir,
            },
        )
        swanlab_run_id = experiment.public.run_id

    save_dir = Path(args.save_dir) / task_name / args.model_kind
    artifact_dir = Path(args.artifact_dir) / task_name / args.model_kind
    train_iter = infinite_loader(train_loader)
    model.train()

    for step in range(args.steps):
        step_start = time.time()
        batch = move_batch(next(train_iter), device)

        if args.model_kind == "camoe":
            phase, critic_alpha = get_phase(step, config)
            optimizer.zero_grad(set_to_none=True)
            result = model(
                batch["input_ids"],
                batch["targets"],
                critic_alpha=critic_alpha,
                training=True,
                uniform=(phase == "prewarm"),
            )
            result["loss_scalar"].backward()
            clip_grad_norm_(expert_params, config.grad_clip)
            optimizer.step()

            settle_results = []
            if phase != "prewarm":
                with torch.no_grad():
                    settle_results = model.settle_all_layers(
                        result["loss"].detach(),
                        token_weight=batch["supervised_mask"].detach(),
                    )

            critic_loss_value = 0.0
            if phase not in ("prewarm", "market_warm") and settle_results and step % config.critic_update_interval == 0:
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss = model.compute_critic_loss(settle_results)
                critic_loss.backward()
                clip_grad_norm_(critic_params, 1.0)
                critic_optimizer.step()
                critic_loss_value = float(critic_loss.detach().item())
        else:
            phase, critic_alpha = "single", 1.0
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["input_ids"], batch["targets"])
            result["loss_scalar"].backward()
            clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            critic_loss_value = 0.0

        train_token_acc, train_exact = masked_metrics(result["logits"], batch["targets"], batch["supervised_mask"])
        if step % args.log_interval == 0:
            logs = {
                "train/loss": float(result["loss_scalar"].detach().item()),
                "train/answer_token_acc": float(train_token_acc.item()),
                "train/exact_match": float(train_exact.item()),
                "train/step_time_sec": float(time.time() - step_start),
            }
            if args.model_kind == "camoe":
                logs["train/critic_loss"] = float(critic_loss_value)
                logs["train/critic_alpha"] = float(critic_alpha)
                logs.update(market_logs_from_batch(model, batch))
            print(
                f"step={step} task={task_name} kind={args.model_kind} phase={phase} "
                f"loss={logs['train/loss']:.4f} acc={logs['train/answer_token_acc']:.4f} "
                f"exact={logs['train/exact_match']:.4f}"
            )
            if HAS_SWANLAB and not args.no_swanlab:
                swanlab.log(logs, step=step)

        if step % args.eval_interval == 0 or step == args.steps - 1:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                args.model_kind,
                critic_alpha=critic_alpha,
                uniform=(args.model_kind == "camoe" and phase == "prewarm"),
            )
            ood_metrics = evaluate_model(
                model,
                ood_loader,
                device,
                args.model_kind,
                critic_alpha=critic_alpha,
                uniform=(args.model_kind == "camoe" and phase == "prewarm"),
            )
            eval_logs = {
                "val/loss": val_metrics["loss"],
                "val/answer_token_acc": val_metrics["answer_token_acc"],
                "val/exact_match": val_metrics["exact_match"],
                "ood/answer_token_acc": ood_metrics["answer_token_acc"],
                "ood/exact_match": ood_metrics["exact_match"],
            }
            for op_name in OPERATION_NAME_BY_ID.values():
                if f"{op_name}/answer_token_acc" in val_metrics:
                    eval_logs[f"val/{op_name}/answer_token_acc"] = val_metrics[f"{op_name}/answer_token_acc"]
                if f"{op_name}/exact_match" in val_metrics:
                    eval_logs[f"val/{op_name}/exact_match"] = val_metrics[f"{op_name}/exact_match"]
                if f"{op_name}/answer_token_acc" in ood_metrics:
                    eval_logs[f"ood/{op_name}/answer_token_acc"] = ood_metrics[f"{op_name}/answer_token_acc"]
                if f"{op_name}/exact_match" in ood_metrics:
                    eval_logs[f"ood/{op_name}/exact_match"] = ood_metrics[f"{op_name}/exact_match"]
            print(
                f"[eval] task={task_name} step={step} val_acc={val_metrics['answer_token_acc']:.4f} "
                f"val_exact={val_metrics['exact_match']:.4f} "
                f"ood_acc={ood_metrics['answer_token_acc']:.4f}"
            )
            val_by_op = []
            ood_by_op = []
            for op_name in OPERATION_NAME_BY_ID.values():
                val_key = f"{op_name}/answer_token_acc"
                ood_key = f"{op_name}/answer_token_acc"
                if val_key in val_metrics:
                    val_by_op.append(f"{op_name}={val_metrics[val_key]:.4f}")
                if ood_key in ood_metrics:
                    ood_by_op.append(f"{op_name}={ood_metrics[ood_key]:.4f}")
            if val_by_op:
                print(f"[eval-by-op] val {' '.join(val_by_op)}")
            if ood_by_op:
                print(f"[eval-by-op] ood {' '.join(ood_by_op)}")
            if args.model_kind == "camoe" and phase != "prewarm":
                render_route_preview(
                    model,
                    preview_batch,
                    artifact_dir,
                    step,
                    task_name,
                    critic_alpha=critic_alpha,
                    uniform=(phase == "prewarm"),
                )
                preview_logs = preview_route_stats(model, preview_batch)
                if preview_logs:
                    for key, value in sorted(preview_logs.items()):
                        print(f"[route-stats] {key}={value:.4f}")
                    eval_logs.update(preview_logs)
            if HAS_SWANLAB and not args.no_swanlab:
                swanlab.log(eval_logs, step=step)

        if step > 0 and step % args.save_interval == 0:
            save_checkpoint(
                save_dir / f"step_{step:06d}.pth",
                model,
                optimizer,
                step,
                config,
                critic_optimizer=critic_optimizer,
                swanlab_run_id=swanlab_run_id,
            )

    save_checkpoint(
        save_dir / "final.pth",
        model,
        optimizer,
        args.steps - 1,
        config,
        critic_optimizer=critic_optimizer,
        swanlab_run_id=swanlab_run_id,
    )


if __name__ == "__main__":
    main()
