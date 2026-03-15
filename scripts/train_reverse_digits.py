"""Train tiny toy-sequence baselines for ROSA and CaMoE."""

from __future__ import annotations

import html
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Literal

import torch
import typer
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from camoe.config import CaMoEConfig, ROSA_BACKEND_CHOICES
from camoe.model import CaMoE_Model
from camoe.expert_rosa import ROSAExpert
from camoe.reverse_baselines import (
    BaselinePureRosaRWKVFFN,
    BaselineTimeMixRosaRWKVFFN,
    BaselineTimeMixRWKVFFN,
)

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
PATTERN_ID = 20
DELAY_ID = 21
REPEAT_ID = 22
RUNMAX_ID = 23
THRESH_ID = 24
BRACKET_ID = 25
MASK_ID = 26
LPAREN_ID = 27
RPAREN_ID = 28
PLUS_ID = 29
MINUS_ID = 30
ADDSUB_ID = 31
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
    PATTERN_ID: "[PATTERN]",
    DELAY_ID: "[DELAY]",
    REPEAT_ID: "[REPEAT]",
    RUNMAX_ID: "[RUNMAX]",
    THRESH_ID: "[THRESH]",
    BRACKET_ID: "[BRACKET]",
    MASK_ID: "[MASK]",
    LPAREN_ID: "(",
    RPAREN_ID: ")",
    PLUS_ID: "+",
    MINUS_ID: "-",
    ADDSUB_ID: "[ADDSUB]",
}
TOKEN_NAMES.update({4 + value: str(value) for value in range(10)})
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
EXPERT_ANSI_PALETTE = [
    "\033[94m",
    "\033[91m",
    "\033[92m",
    "\033[95m",
    "\033[96m",
    "\033[33m",
    "\033[37m",
    "\033[90m",
]
EXPERT_HTML_PALETTE = [
    "#62b0ff",
    "#ff6b6b",
    "#55d88a",
    "#ff9ff3",
    "#63e6be",
    "#ffd166",
    "#f8f9fa",
    "#adb5bd",
]
OPERATION_NAME_BY_ID = {
    0: "reverse_digits",
    1: "copy_digits",
    2: "parity_digits",
    3: "cumsum_mod10",
    4: "majority_vote",
    5: "count_ones",
    6: "pattern_complete",
    7: "delayed_copy",
    8: "first_repeat",
    9: "running_max",
    10: "sum_threshold",
    11: "bracket_depth",
    12: "addsub_40",
    13: "pattern_continue",
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
        if "reward_critic" in name:
            critic_params.append(param)
        else:
            expert_params.append(param)
    return expert_params, critic_params


def _compute_market_alpha(step: int, config: CaMoEConfig) -> float:
    del step, config
    return 1.0


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


def get_phase(step: int, config: CaMoEConfig) -> tuple[str, float]:
    if step < config.uniform_warmup_steps:
        return "uniform_warmup", 1.0
    return "full_market", 1.0


def filter_dataset_by_operation(dataset: Dataset, operation_name: str | None) -> Dataset:
    if not operation_name:
        return dataset
    return dataset.filter(
        lambda row: (row.get("operation") or row.get("task")) == operation_name
    )


def infer_max_sequence_length(*datasets: Dataset) -> int:
    max_length = 0
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        max_length = max(
            max_length,
            max(len(token_ids) for token_ids in dataset["input_ids"]),
        )
    return max_length


def build_config(options: SimpleNamespace, seq_len: int) -> CaMoEConfig:
    return CaMoEConfig(
        vocab_size=32,
        dim=options.dim,
        n_layers=options.n_layers,
        n_heads=options.n_heads,
        n_experts=2,
        n_deepembed_experts=options.n_deepembed_experts,
        n_slim_deepembed_experts=options.n_slim_deepembed_experts,
        n_rosa_experts=1,
        ffn_expand=4,
        tie_weights=True,
        rosa_backend=options.rosa_backend,
        rosa_hard_backend=options.rosa_hard_backend,
        rosa_hard_switch_step=options.rosa_hard_switch_step,
        rosa_bits=options.rosa_bits,
        slim_rosa_heads=options.slim_rosa_heads,
        rosa_truncation_length=options.rosa_truncation_length,
        rosa_native_mode=bool(options.rosa_native_mode),
        rosa_use_gate=bool(options.rosa_use_gate),
        deepembed_mode=options.deepembed_mode,
        deepembed_expand=options.deepembed_expand,
        slim_deepembed_rank=options.slim_deepembed_rank,
        auction_noise_std=options.auction_noise_std,
        routing_noise_std=options.routing_noise_std,
        exploration_epsilon=options.exploration_epsilon,
        bet_fraction=options.bet_fraction,
        price_lr=options.price_lr,
        price_temperature=options.price_temperature,
        liquidity_floor=options.liquidity_floor,
        market_alpha_start=options.market_alpha_start,
        market_alpha_end=options.market_alpha_end,
        routing_ste=bool(options.routing_ste),
        ste_temperature_start=options.ste_temperature_start,
        ste_temperature_mid=options.ste_temperature_mid,
        ste_temperature_end=options.ste_temperature_end,
        ste_midpoint_steps=options.ste_midpoint_steps,
        ste_anneal_steps=options.ste_anneal_steps,
        market_ramp_steps=options.market_ramp_steps,
        enable_compile=False,
        enable_gradient_checkpointing=False,
        expert_capital_init=1.0,
        critic_capital_init=0.5,
        ema_decay=0.95,
        capital_floor=0.25,
        capital_ceiling=options.capital_ceiling,
        depreciation=options.depreciation,
        critic_hidden_dim=None,
        reward_hidden_dim=options.reward_hidden_dim,
        critic_update_interval=options.critic_update_interval,
        critic_lr=options.critic_lr,
        reward_scale=options.reward_scale,
        reward_eps=options.reward_eps,
        routing_entropy_reg=options.routing_entropy_reg,
        critic_shadow_prewarm=bool(options.critic_shadow_prewarm),
        critic_shadow_market=bool(options.critic_shadow_market),
        uniform_warmup_steps=options.uniform_warmup_steps,
        prewarm_steps=options.prewarm_steps,
        market_warmup_steps=options.market_warmup_steps,
        critic_warmup_steps=options.critic_warmup_steps,
        lr=options.lr,
        batch_size=options.batch_size,
        seq_len=seq_len,
        total_steps=options.steps,
        grad_clip=1.0,
        ignore_index=IGNORE_INDEX,
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_kind: str,
    critic_alpha: float = 1.0,
    ste_temperature: float = 0.3,
    uniform: bool = False,
    market_weight: float = 1.0,
    current_step: int | None = None,
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
                    ste_temperature=ste_temperature,
                    training=False,
                    uniform=uniform,
                    market_weight=market_weight,
                    current_step=current_step,
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


def build_expert_styles(expert_types: list[str]) -> list[dict[str, str]]:
    styles = []
    total_counts: dict[str, int] = {}
    for expert_type in expert_types:
        total_counts[expert_type] = total_counts.get(expert_type, 0) + 1
    type_counts: dict[str, int] = {}
    for expert_idx, expert_type in enumerate(expert_types):
        count = type_counts.get(expert_type, 0)
        type_counts[expert_type] = count + 1
        suffix = f"#{count}" if total_counts[expert_type] > 1 else ""
        styles.append(
            {
                "label": f"{expert_type}{suffix}",
                "ansi": EXPERT_ANSI_PALETTE[expert_idx % len(EXPERT_ANSI_PALETTE)],
                "html": EXPERT_HTML_PALETTE[expert_idx % len(EXPERT_HTML_PALETTE)],
            }
        )
    return styles


def colorize_token(token: str, winner: int, is_answer: bool, expert_styles: list[dict[str, str]]) -> str:
    if token == "[SEP]":
        return f"{ANSI_YELLOW}{token}{ANSI_RESET}"
    color = expert_styles[winner]["ansi"] if 0 <= winner < len(expert_styles) else ANSI_RESET
    decorated = f"{token}" if is_answer else f"({token})"
    return f"{color}{decorated}{ANSI_RESET}"


def render_route_preview(
    model: CaMoE_Model,
    batch: dict[str, torch.Tensor],
    artifact_dir: Path,
    step: int,
    task_name: str,
    critic_alpha: float,
    ste_temperature: float,
    uniform: bool,
    market_weight: float,
    current_step: int | None = None,
) -> None:
    model.eval()
    with torch.no_grad():
        model(
            batch["input_ids"],
            batch["targets"],
            critic_alpha=critic_alpha,
            ste_temperature=ste_temperature,
            training=False,
            uniform=uniform,
            market_weight=market_weight,
            current_step=current_step,
        )
    diagnostics = model.get_market_diagnostics()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    html_path = artifact_dir / f"step_{step:06d}.html"

    html_parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{font-family:Consolas,monospace;background:#111;color:#eee;padding:16px;}",
        ".sep{color:#ffd166;font-weight:700;} .answer{font-weight:700;} .prefix{opacity:0.65;}",
        ".token{display:inline-block;margin:0 2px 6px 0;padding:2px 5px;border-radius:4px;background:#1b1b1b;}",
        ".legend{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px;}",
        ".legend-item{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border:1px solid #333;border-radius:999px;background:#181818;}",
        ".legend-swatch{width:10px;height:10px;border-radius:999px;display:inline-block;}",
        ".op-sep{margin-top:18px;padding-top:10px;border-top:1px solid #333;}",
        "table{border-collapse:collapse;margin:12px 0;} td,th{border:1px solid #333;padding:4px 6px;} h2,h3,h4{margin:16px 0 8px;}",
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
        expert_styles = build_expert_styles(list(entry["expert_types"]))
        capitals = entry["capitals"].cpu().tolist()
        cache = entry["cache"]
        winners = cache["winners"].cpu()
        expected_profit = cache.get("expected_profit")
        expected_profit = expected_profit.cpu() if expected_profit is not None else None
        html_parts.append(f"<h2>Layer {layer_idx} | {html.escape(market_name)}</h2>")
        #print(f"[route-preview] task={task_name} layer={layer_idx} market={market_name}")
        legend_chunks = []
        for expert_idx, style in enumerate(expert_styles):
            capital_value = float(capitals[expert_idx]) if expert_idx < len(capitals) else 0.0
            legend_chunks.append(
                "<span class=\"legend-item\">"
                f"<span class=\"legend-swatch\" style=\"background:{html.escape(style['html'])};\"></span>"
                f"{expert_idx}: {html.escape(style['label'])} | cap={capital_value:.3f}"
                "</span>"
            )
        html_parts.append("<div class=\"legend\">" + "".join(legend_chunks) + "</div>")
        #print(
        #    "  legend:",
        #    ", ".join(
        #        f"{expert_idx}={style['label']}(cap={float(capitals[expert_idx]):.3f})"
        #        for expert_idx, style in enumerate(expert_styles)
        #    ),
        #)

        last_op_name = None
        for sample_idx in range(targets_cpu.size(0)):
            tokens = decode_display_stream(input_ids_cpu[sample_idx], targets_cpu[sample_idx])
            winners_row = winners[sample_idx].tolist()
            answer_row = answer_mask_cpu[sample_idx].tolist()
            valid_row = valid_mask_cpu[sample_idx].tolist()
            if expected_profit is None:
                selected_scores = None
            else:
                selected_scores = expected_profit[
                    sample_idx, torch.arange(winners.size(1)), winners[sample_idx]
                ].tolist()
            op_name = OPERATION_NAME_BY_ID.get(int(op_ids_cpu[sample_idx].item()), "unknown")
            if op_name != last_op_name:
                html_parts.append(f"<div class=\"op-sep\"><h3>Operation: {html.escape(op_name)}</h3></div>")
                last_op_name = op_name

            html_tokens = []
            rows = []
            bid_iter = selected_scores if selected_scores is not None else [None] * len(tokens)
            for pos, (token, winner, is_answer, is_valid, bid_value) in enumerate(
                zip(tokens, winners_row, answer_row, valid_row, bid_iter)
            ):
                if not is_valid:
                    continue
                winner_idx = int(winner)
                token_classes = ["token", "answer" if is_answer else "prefix"]
                style_attr = ""
                if token == "[SEP]":
                    token_classes.append("sep")
                elif 0 <= winner_idx < len(expert_styles):
                    style_attr = f" style=\"color:{html.escape(expert_styles[winner_idx]['html'])};\""
                html_tokens.append(
                    f"<span class=\"{' '.join(token_classes)}\"{style_attr}>{html.escape(token)}</span>"
                )
                bid_display = "N/A" if bid_value is None else f"{float(bid_value):.4f}"
                winner_label = expert_styles[winner_idx]["label"] if 0 <= winner_idx < len(expert_styles) else "unknown"
                rows.append(
                    "<tr>"
                    f"<td>{pos}</td><td>{html.escape(token)}</td><td>{winner_idx}</td>"
                    f"<td>{html.escape(winner_label)}</td>"
                    f"<td>{bid_display}</td><td>{int(bool(is_answer))}</td>"
                    "</tr>"
                )

            html_parts.append(f"<h4>Sample {sample_idx} | {html.escape(op_name)}</h4>")
            html_parts.append(f"<div>{' '.join(html_tokens)}</div>")
            html_parts.append(
                    "<table><tr><th>pos</th><th>token</th><th>winner_id</th><th>winner</th><th>score</th><th>answer</th></tr>"
                )
            html_parts.extend(rows)
            html_parts.append("</table>")

    html_parts.append("</body></html>")
    html_path.write_text("".join(html_parts), encoding="utf-8")
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


def _iter_rosa_experts(model: torch.nn.Module) -> list[tuple[str, ROSAExpert]]:
    experts: list[tuple[str, ROSAExpert]] = []
    if isinstance(model, CaMoE_Model):
        for layer_idx, block in enumerate(model.blocks):
            if block.rosa_expert is not None:
                experts.append((f"layer_{layer_idx}", block.rosa_expert))
        return experts

    rosa_layers = getattr(model, "rosa_layers", None)
    if rosa_layers is None:
        return experts
    for layer_idx, expert in enumerate(rosa_layers):
        if isinstance(expert, ROSAExpert):
            experts.append((f"layer_{layer_idx}", expert))
    return experts


def emit_inner_language_log(
    *,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    step: int,
    task_name: str,
    model_kind: str,
    artifact_dir: Path,
    critic_alpha: float,
    ste_temperature: float,
    uniform: bool,
    market_weight: float,
    current_step: int | None = None,
) -> dict[str, float]:
    rosa_experts = _iter_rosa_experts(model)
    if not rosa_experts:
        return {}

    captured_inputs: dict[str, torch.Tensor] = {}
    handles = []
    for label, expert in rosa_experts:
        def _hook(module, inputs, label=label):
            if inputs:
                captured_inputs[label] = inputs[0].detach()

        handles.append(expert.register_forward_pre_hook(_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        if model_kind == "camoe":
            model(
                batch["input_ids"],
                batch["targets"],
                critic_alpha=critic_alpha,
                ste_temperature=ste_temperature,
                training=False,
                uniform=uniform,
                market_weight=market_weight,
                current_step=current_step,
            )
        else:
            model(batch["input_ids"], batch["targets"])
    for handle in handles:
        handle.remove()
    model.train(was_training)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    text_path = artifact_dir / f"step_{step:06d}_inner_language.txt"
    lines = [f"task={task_name} step={step} model_kind={model_kind}"]
    logs: dict[str, float] = {}

    for label, expert in rosa_experts:
        x_in = captured_inputs.get(label)
        if x_in is None:
            continue
        summary = expert.summarize_symbol_language(x_in, valid_mask=batch["valid_mask"])
        prefix = f"inner_lang/{label}"
        logs[f"{prefix}/q_bit_balance"] = float(summary["q_bit_balance"])
        logs[f"{prefix}/q_margin_mean"] = float(summary["q_margin_mean"])
        logs[f"{prefix}/q_vocab_mean"] = float(summary["q_vocab_mean"])
        logs[f"{prefix}/q_entropy_mean"] = float(summary["q_entropy_mean"])
        logs[f"{prefix}/q_repeat_rate_mean"] = float(summary["q_repeat_rate_mean"])

        print(
            f"[inner-lang][step {step}] {label} "
            f"backend={summary['backend']} bits={summary['bits_per_symbol']} heads={summary['heads']} "
            f"q_vocab_mean={float(summary['q_vocab_mean']):.2f} "
            f"q_entropy_mean={float(summary['q_entropy_mean']):.3f} "
            f"q_repeat_rate={float(summary['q_repeat_rate_mean']):.3f} "
            f"q_bit_balance={float(summary['q_bit_balance']):.3f} "
            f"q_margin_mean={float(summary['q_margin_mean']):.3f}"
        )
        lines.append(
            f"[{label}] backend={summary['backend']} bits={summary['bits_per_symbol']} heads={summary['heads']} "
            f"valid_tokens={summary['valid_tokens']} q_vocab_mean={float(summary['q_vocab_mean']):.4f} "
            f"q_entropy_mean={float(summary['q_entropy_mean']):.4f} "
            f"q_repeat_rate_mean={float(summary['q_repeat_rate_mean']):.4f} "
            f"q_bit_balance={float(summary['q_bit_balance']):.4f} "
            f"q_margin_mean={float(summary['q_margin_mean']):.4f}"
        )

        for head_name, items in summary["top_q_symbols"].items():
            joined = ", ".join(items)
            print(f"[inner-lang][step {step}] {label} {head_name} top_q={joined}")
            lines.append(f"[{label}] {head_name} top_q={joined}")

        for stream_name, stream_map in (
            ("q", summary["sample0_q"]),
            ("k", summary["sample0_k"]),
            ("v", summary["sample0_v"]),
        ):
            for head_name, symbols in stream_map.items():
                joined = " ".join(symbols)
                print(f"[inner-lang][step {step}] {label} sample0_{stream_name}_{head_name}={joined}")
                lines.append(f"[{label}] sample0_{stream_name}_{head_name}={joined}")

    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return logs


def market_logs_from_batch(
    model: CaMoE_Model,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    metrics = model.market_metrics(token_mask=batch["supervised_mask"], valid_mask=batch["valid_mask"])
    logs: dict[str, float] = {}
    for layer_idx, block in enumerate(model.blocks):
        market_specs: list[tuple[str, list[str]]] = []
        if block.has_sequence_market:
            market_specs.append(("sequence", [expert.expert_type for expert in block.sequence_experts()]))
        market_specs.append(("ffn", [expert.expert_type for expert in block.experts]))

        for market_name, expert_types in market_specs:
            prefix = f"{market_name}/layer_{layer_idx}"
            logs[f"market/{market_name}/layer_{layer_idx}/wallet_mean"] = metrics.get(
                f"{prefix}/wallet_mean",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/wallet_gini"] = metrics.get(
                f"{prefix}/wallet_gini",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/price_max"] = metrics.get(
                f"{prefix}/price_max",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/routing_entropy"] = metrics.get(
                f"{prefix}/routing_entropy",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/routing_entropy_answer"] = metrics.get(
                f"{prefix}/routing_entropy_answer",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/pred_reward_mean"] = metrics.get(
                f"{prefix}/pred_reward_mean",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/expected_profit_mean"] = metrics.get(
                f"{prefix}/expected_profit_mean",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/realized_reward_mean"] = metrics.get(
                f"{prefix}/realized_reward_mean",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/token_profit_mean"] = metrics.get(
                f"{prefix}/token_profit_mean",
                0.0,
            )
            logs[f"market/{market_name}/layer_{layer_idx}/exploration_rate"] = metrics.get(
                f"{prefix}/exploration_rate",
                0.0,
            )

            for expert_idx, expert_type in enumerate(expert_types):
                expert_prefix = f"{prefix}/expert_{expert_idx}"
                expert_label = (
                    expert_type
                    if expert_types.count(expert_type) == 1
                    else f"{expert_type}_{expert_idx}"
                )
                logs[f"market/{market_name}/layer_{layer_idx}/wallet_{expert_label}"] = metrics.get(
                    f"{expert_prefix}/wallet",
                    0.0,
                )
                logs[f"market/{market_name}/layer_{layer_idx}/price_{expert_label}"] = metrics.get(
                    f"{expert_prefix}/price",
                    0.0,
                )
                logs[f"market/{market_name}/layer_{layer_idx}/q_{expert_label}"] = metrics.get(
                    f"{expert_prefix}/q",
                    0.0,
                )
                logs[f"market/{market_name}/layer_{layer_idx}/winner_share_{expert_label}"] = metrics.get(
                    f"{expert_prefix}/winner_share",
                    0.0,
                )
                logs[f"market/{market_name}/layer_{layer_idx}/winner_share_{expert_label}_answer"] = metrics.get(
                    f"{expert_prefix}/winner_share_answer",
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
                market_specs = []
                if block.has_sequence_market:
                    market_specs.append(("sequence", [expert.expert_type for expert in block.sequence_experts()]))
                market_specs.append(("ffn", [expert.expert_type for expert in block.experts]))

                for market_name, expert_types in market_specs:
                    prefix = f"{market_name}/layer_{layer_idx}"
                    logs[f"market/{op_name}/{market_name}/layer_{layer_idx}/routing_entropy_answer"] = op_metrics.get(
                        f"{prefix}/routing_entropy_answer",
                        0.0,
                    )
                    for expert_idx, expert_type in enumerate(expert_types):
                        expert_label = (
                            expert_type
                            if expert_types.count(expert_type) == 1
                            else f"{expert_type}_{expert_idx}"
                        )
                        logs[
                            f"market/{op_name}/{market_name}/layer_{layer_idx}/winner_share_{expert_label}_answer"
                        ] = op_metrics.get(
                            f"{prefix}/expert_{expert_idx}/winner_share_answer",
                            0.0,
                        )
    return logs


def _format_tensor_list(tensor: torch.Tensor | None, digits: int = 4) -> list[float] | None:
    if tensor is None:
        return None
    flat = tensor.detach().float().cpu().reshape(-1).tolist()
    return [round(float(value), digits) for value in flat]


def _set_force_winner(
    model: CaMoE_Model,
    *,
    sequence_idx: int | None,
    ffn_idx: int | None,
) -> None:
    if model.sequence_capital_manager is not None:
        model.sequence_capital_manager.force_winner = sequence_idx
    model.ffn_capital_manager.force_winner = ffn_idx


def emit_debug_step(
    *,
    step: int,
    model: CaMoE_Model,
    batch: dict[str, torch.Tensor],
    result: dict[str, torch.Tensor],
    settle_results: list[dict[str, torch.Tensor | int | str | dict[str, torch.Tensor]]],
    phase: str,
    ste_temperature: float,
    exploration_epsilon: float,
    critic_loss_value: float,
) -> None:
    supervised = float(batch["supervised_mask"].sum().item())
    valid = float(batch["valid_mask"].sum().item())
    print(
        f"[debug][step {step}] phase={phase} "
        f"loss_scalar={float(result['loss_scalar'].detach().item()):.6f} "
        f"loss_main={float(result.get('loss_main_scalar', result['loss_scalar']).detach().item()):.6f} "
        f"critic_loss={float(critic_loss_value):.6f} "
        f"ste_temperature={float(ste_temperature):.4f} "
        f"exploration_epsilon={float(exploration_epsilon):.4f} "
        f"supervised_tokens={supervised:.0f} valid_tokens={valid:.0f}"
    )

    settle_map = {(int(item["layer"]), str(item["market"])): item for item in settle_results}
    diagnostics = model.get_market_diagnostics()
    for entry in diagnostics:
        layer_idx = int(entry["layer"])
        market_name = str(entry.get("market", "ffn"))
        expert_types = list(entry.get("expert_types", []))
        caps = entry.get("capitals")
        prices = entry.get("prices")
        q = entry.get("q")
        loss_ema = entry.get("loss_ema")
        cache = entry.get("cache", {})
        winners = cache.get("winners")
        pred_reward = cache.get("pred_reward")
        expected_profit = cache.get("expected_profit")
        exploration_mask = cache.get("exploration_mask")
        if winners is None:
            continue

        winners_cpu = winners.detach().cpu()
        winner_hist = torch.bincount(winners_cpu.reshape(-1), minlength=len(expert_types)).float()
        winner_hist = winner_hist / winner_hist.sum().clamp(min=1.0)
        preview_steps = min(12, winners_cpu.size(1))
        preview_winners = winners_cpu[0, :preview_steps].tolist()

        print(
            f"[debug][step {step}] L{layer_idx}/{market_name} "
            f"experts={expert_types} "
            f"loss_ema={None if loss_ema is None else round(float(loss_ema.detach().item()), 6)}"
        )
        print(
            f"[debug][step {step}] L{layer_idx}/{market_name} "
            f"caps={_format_tensor_list(caps)} "
            f"prices={_format_tensor_list(prices)} "
            f"q={_format_tensor_list(q)} "
            f"winner_hist={_format_tensor_list(winner_hist)}"
        )

        if pred_reward is not None:
            chosen_pred = torch.gather(pred_reward, dim=-1, index=winners.unsqueeze(-1)).squeeze(-1)
            pred_preview = chosen_pred[0, :preview_steps].detach().cpu().tolist()
            pred_preview = [round(float(value), 4) for value in pred_preview]
        else:
            pred_preview = None

        if expected_profit is not None:
            chosen_profit = torch.gather(expected_profit, dim=-1, index=winners.unsqueeze(-1)).squeeze(-1)
            profit_preview = chosen_profit[0, :preview_steps].detach().cpu().tolist()
            profit_preview = [round(float(value), 4) for value in profit_preview]
        else:
            profit_preview = None

        exploration_rate = (
            None
            if exploration_mask is None
            else round(float(exploration_mask.float().mean().detach().cpu().item()), 4)
        )
        print(
            f"[debug][step {step}] L{layer_idx}/{market_name} "
            f"sample0_winners={preview_winners} "
            f"sample0_pred_reward={pred_preview} "
            f"sample0_expected_profit={profit_preview} "
            f"exploration_rate={exploration_rate}"
        )

        settle = settle_map.get((layer_idx, market_name))
        if settle is not None:
            print(
                f"[debug][step {step}] L{layer_idx}/{market_name} "
                f"profit={_format_tensor_list(settle.get('profit'))} "
                f"avg_reward={_format_tensor_list(settle.get('avg_reward'))} "
                f"winner_share={_format_tensor_list(settle.get('winner_share'))}"
            )


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


def main(
    data_dir: str = typer.Option("data/reverse_digits"),
    task_filter: str = typer.Option("reverse_digits"),
    mode: Literal["normal", "debug", "single-debug"] = typer.Option("normal"),
    debug_sequence_expert_idx: int = typer.Option(0),
    debug_ffn_expert_idx: int = typer.Option(0),
    model_kind: Literal["timemix_rosa_ffn", "timemix_ffn", "pure_rosa_ffn", "camoe"] = typer.Option("camoe"),
    device: str = typer.Option("cuda"),
    batch_size: int = typer.Option(128),
    eval_batch_size: int = typer.Option(256),
    steps: int = typer.Option(10000),
    eval_interval: int = typer.Option(250),
    log_interval: int = typer.Option(50),
    inner_language_log_interval: int = typer.Option(0),
    save_interval: int = typer.Option(1000),
    num_workers: int = typer.Option(0),
    lr: float = typer.Option(1e-3),
    critic_lr: float = typer.Option(3e-4),
    weight_decay: float = typer.Option(0.01),
    critic_update_interval: int = typer.Option(8),
    bet_fraction: float = typer.Option(0.05),
    price_lr: float = typer.Option(0.02),
    price_temperature: float = typer.Option(1.0),
    liquidity_floor: float = typer.Option(0.02),
    reward_scale: float = typer.Option(5.0),
    reward_eps: float = typer.Option(1e-8),
    reward_hidden_dim: int | None = typer.Option(None),
    uniform_warmup_steps: int = typer.Option(1500),
    market_ramp_steps: int = typer.Option(1000),
    routing_noise_std: float = typer.Option(0.05),
    exploration_epsilon: float = typer.Option(0.02),
    seed: int = typer.Option(42),
    n_layers: int = typer.Option(2),
    dim: int = typer.Option(64),
    n_heads: int = typer.Option(4),
    n_deepembed_experts: int = typer.Option(0),
    n_slim_deepembed_experts: int = typer.Option(0),
    deepembed_mode: Literal["1x", "4x"] = typer.Option("1x"),
    deepembed_expand: int = typer.Option(4),
    slim_deepembed_rank: int = typer.Option(32),
    slim_rosa_heads: int = typer.Option(8),
    rosa_backend: str = typer.Option(
        "hard_symbolic_multibit",
        help=f"Canonical backends: {', '.join(ROSA_BACKEND_CHOICES)}. Old aliases still work.",
    ),
    rosa_hard_backend: str | None = typer.Option(
        None,
        help=f"Canonical backends: {', '.join(ROSA_BACKEND_CHOICES)}. Old aliases still work.",
    ),
    rosa_hard_switch_step: int | None = typer.Option(None),
    rosa_bits: int = typer.Option(8),
    rosa_truncation_length: int = typer.Option(8),
    rosa_native_mode: bool = typer.Option(False, "--rosa-native-mode/--no-rosa-native-mode"),
    rosa_use_gate: bool = typer.Option(True, "--rosa-use-gate/--no-rosa-use-gate"),
    auction_noise_std: float = typer.Option(0.05),
    market_alpha_start: float = typer.Option(0.0),
    market_alpha_end: float = typer.Option(1.0),
    routing_entropy_reg: float = typer.Option(0.0),
    critic_shadow_prewarm: bool = typer.Option(True, "--critic-shadow-prewarm/--no-critic-shadow-prewarm"),
    critic_shadow_market: bool = typer.Option(True, "--critic-shadow-market/--no-critic-shadow-market"),
    routing_ste: bool = typer.Option(True, "--routing-ste/--no-routing-ste"),
    ste_temperature_start: float = typer.Option(2.0),
    ste_temperature_mid: float = typer.Option(1.0),
    ste_temperature_end: float = typer.Option(0.3),
    ste_midpoint_steps: int = typer.Option(1500),
    ste_anneal_steps: int = typer.Option(4000),
    depreciation: float = typer.Option(1e-3),
    capital_ceiling: float = typer.Option(10.0),
    prewarm_steps: int = typer.Option(1500),
    market_warmup_steps: int = typer.Option(1500),
    critic_warmup_steps: int = typer.Option(1000),
    save_dir: str = typer.Option("checkpoints/reverse_digits"),
    artifact_dir: str = typer.Option("artifacts/reverse_digits"),
    resume: str = typer.Option(""),
    stop_on_val_exact: bool = typer.Option(False, "--stop-on-val-exact/--no-stop-on-val-exact"),
    stop_on_val_exact_threshold: float = typer.Option(0.0),
    max_eval_length: int = typer.Option(20),
    no_swanlab: bool = typer.Option(False, "--no-swanlab/--swanlab"),
) -> None:
    args = SimpleNamespace(**locals())

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    task_name, train_ds, val_ds, ood_ds = load_splits(args.data_dir)
    train_ds = filter_dataset_by_operation(train_ds, args.task_filter)
    val_ds = filter_dataset_by_operation(val_ds, args.task_filter)
    ood_ds = filter_dataset_by_operation(ood_ds, args.task_filter)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(ood_ds) == 0:
        raise ValueError(f"task_filter={args.task_filter!r} produced an empty split from {args.data_dir}.")
    if args.task_filter:
        task_name = args.task_filter
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

    detected_seq_len = pad_to_chunk(infer_max_sequence_length(train_ds, val_ds, ood_ds))
    resume_payload: dict | None = None
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu")
        config = CaMoEConfig.from_mapping(resume_payload["config"])
        config.seq_len = detected_seq_len
        config.total_steps = args.steps
        config.batch_size = args.batch_size
        config.lr = args.lr
        config.critic_lr = args.critic_lr
    else:
        config = build_config(args, seq_len=detected_seq_len)
    print(
        f"[config] task={task_name} detected_seq_len={detected_seq_len} "
        f"(from train/val/ood max sample length)"
    )
    if args.mode == "single-debug" and args.model_kind != "camoe":
        raise ValueError("mode=single-debug requires --model_kind camoe.")
    if args.model_kind == "camoe":
        model: torch.nn.Module = CaMoE_Model(config).to(device)
        expert_params, critic_params = split_optim_params(model)
        optimizer = torch.optim.AdamW(expert_params, lr=args.lr, weight_decay=args.weight_decay)
        critic_optimizer = torch.optim.AdamW(critic_params, lr=args.critic_lr, weight_decay=args.weight_decay)
        sequence_debug_idx = None
        ffn_debug_idx = None
        if args.mode == "single-debug":
            sequence_debug_idx = max(0, min(args.debug_sequence_expert_idx, 1))
            ffn_debug_idx = max(0, min(args.debug_ffn_expert_idx, config.total_ffn_experts - 1))
        _set_force_winner(
            model,
            sequence_idx=sequence_debug_idx,
            ffn_idx=ffn_debug_idx,
        )
    elif args.model_kind == "timemix_rosa_ffn":
        model = BaselineTimeMixRosaRWKVFFN(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        critic_optimizer = None
    elif args.model_kind == "timemix_ffn":
        model = BaselineTimeMixRWKVFFN(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        critic_optimizer = None
    else:
        model = BaselinePureRosaRWKVFFN(config).to(device)
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
    start_step = 0

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        if critic_optimizer is not None and "critic_optimizer" in resume_payload:
            critic_optimizer.load_state_dict(resume_payload["critic_optimizer"])
        start_step = int(resume_payload.get("step", -1)) + 1
        swanlab_run_id = resume_payload.get("swanlab_run_id", swanlab_run_id)
        print(f"[resume] path={resume_path} start_step={start_step}")
        if start_step >= args.steps:
            raise ValueError(
                f"Resume checkpoint already reached step={start_step - 1}, "
                f"which is not less than requested --steps={args.steps}."
            )

    final_step = start_step - 1
    stop_reason: str | None = None
    for step in range(start_step, args.steps):
        step_start = time.time()
        batch = move_batch(next(train_iter), device)

        if args.model_kind == "camoe":
            phase, critic_alpha = get_phase(step, config)
            uniform = phase == "uniform_warmup"
            ste_temperature = _compute_ste_temperature(step, config)
            dynamic_eps = 0.0 if args.mode == "single-debug" else _compute_exploration_epsilon(step, config)
            market_weight = _compute_market_weight(step, config)
            _set_exploration_epsilon(model, dynamic_eps)
            optimizer.zero_grad(set_to_none=True)
            result = model(
                batch["input_ids"],
                batch["targets"],
                critic_alpha=critic_alpha,
                ste_temperature=ste_temperature,
                training=True,
                uniform=uniform,
                market_weight=market_weight,
                current_step=step,
            )
            result["loss_scalar"].backward()
            clip_grad_norm_(expert_params, config.grad_clip)
            optimizer.step()

            with torch.no_grad():
                settle_results = model.settle_all_layers(
                    result["loss"].detach(),
                    token_weight=batch["supervised_mask"].detach(),
                    update_state=not uniform,
                )

            critic_loss_value = 0.0
            if settle_results and not uniform:
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss = model.compute_critic_loss(
                    settle_results,
                    critic_alpha=critic_alpha,
                    token_weight=(batch["supervised_mask"] * batch["valid_mask"]).detach(),
                )
                critic_loss.backward()
                clip_grad_norm_(critic_params, 1.0)
                critic_optimizer.step()
                critic_loss_value = float(critic_loss.detach().item())
            if args.mode in {"debug", "single-debug"}:
                emit_debug_step(
                    step=step,
                    model=model,
                    batch=batch,
                    result=result,
                    settle_results=settle_results,
                    phase=phase,
                    ste_temperature=ste_temperature,
                    exploration_epsilon=dynamic_eps,
                    critic_loss_value=critic_loss_value,
                )
        else:
            phase, critic_alpha = "single", 1.0
            ste_temperature = config.ste_temperature_end
            market_weight = 1.0
            optimizer.zero_grad(set_to_none=True)
            result = model(batch["input_ids"], batch["targets"])
            result["loss_scalar"].backward()
            clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            critic_loss_value = 0.0
            if args.mode in {"debug", "single-debug"}:
                print(
                    f"[debug][step {step}] phase={phase} "
                    f"loss_scalar={float(result['loss_scalar'].detach().item()):.6f}"
                )

        train_token_acc, train_exact = masked_metrics(result["logits"], batch["targets"], batch["supervised_mask"])
        if step % args.log_interval == 0:
            logs = {
                "train/loss": float(result["loss_scalar"].detach().item()),
                "train/loss_main": float(result.get("loss_main_scalar", result["loss_scalar"]).detach().item()),
                "train/answer_token_acc": float(train_token_acc.item()),
                "train/exact_match": float(train_exact.item()),
                "train/step_time_sec": float(time.time() - step_start),
            }
            if args.model_kind == "camoe":
                logs["train/critic_loss"] = float(critic_loss_value)
                logs["train/ste_temperature"] = float(ste_temperature)
                logs["train/exploration_epsilon"] = float(dynamic_eps)
                logs["train/market_weight"] = float(market_weight)
                if model.blocks and model.blocks[0].rosa_expert is not None:
                    logs["train/rosa_using_hard_backend"] = float(
                        model.blocks[0].rosa_expert.effective_backend(step) == (config.rosa_hard_backend or "")
                    )
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
                ste_temperature=ste_temperature,
                uniform=(phase == "uniform_warmup"),
                market_weight=market_weight,
                current_step=step,
            )
            ood_metrics = evaluate_model(
                model,
                ood_loader,
                device,
                args.model_kind,
                critic_alpha=critic_alpha,
                ste_temperature=ste_temperature,
                uniform=(phase == "uniform_warmup"),
                market_weight=market_weight,
                current_step=step,
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
            if args.model_kind == "camoe":
                render_route_preview(
                    model,
                    preview_batch,
                    artifact_dir,
                    step,
                    task_name,
                    critic_alpha=critic_alpha,
                    ste_temperature=ste_temperature,
                    uniform=(phase == "uniform_warmup"),
                    market_weight=market_weight,
                    current_step=step,
                )
                preview_logs = preview_route_stats(model, preview_batch)
                if preview_logs:
                    for key, value in sorted(preview_logs.items()):
                        #print(f"[route-stats] {key}={value:.4f}")
                        pass
                    eval_logs.update(preview_logs)
            if (
                args.inner_language_log_interval > 0
                and (step % args.inner_language_log_interval == 0 or step == args.steps - 1)
            ):
                inner_logs = emit_inner_language_log(
                    model=model,
                    batch=preview_batch,
                    step=step,
                    task_name=task_name,
                    model_kind=args.model_kind,
                    artifact_dir=artifact_dir,
                    critic_alpha=critic_alpha,
                    ste_temperature=ste_temperature,
                    uniform=(phase == "uniform_warmup"),
                    market_weight=market_weight,
                    current_step=step,
                )
                eval_logs.update(inner_logs)
            if HAS_SWANLAB and not args.no_swanlab:
                swanlab.log(eval_logs, step=step)
            if args.stop_on_val_exact and val_metrics["exact_match"] >= args.stop_on_val_exact_threshold:
                stop_reason = f"val_exact={val_metrics['exact_match']:.4f}"
                final_step = step
                print(f"[early-stop] step={step} reason={stop_reason}")
                break

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
        final_step = step

    save_checkpoint(
        save_dir / "final.pth",
        model,
        optimizer,
        final_step,
        config,
        critic_optimizer=critic_optimizer,
        swanlab_run_id=swanlab_run_id,
    )
    if stop_reason is not None:
        print(f"[done] stopped early at step={final_step} ({stop_reason})")


if __name__ == "__main__":
    typer.run(main)
