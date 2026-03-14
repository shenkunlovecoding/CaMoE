from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from soft_rosa import hard_rosa_reference, soft_rosa_forward, symbols_to_embeddings


@dataclass
class MatchMetrics:
    alpha: float
    gamma: float
    match_position_acc: float
    matched_token_acc: float
    output_mse: float
    matched_positions: int


def _format_vector(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def build_toy_sequence() -> torch.Tensor:
    # Repeating patterns ensure there are many meaningful suffix matches.
    seq = [0, 1, 0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 4]
    return torch.tensor([seq], dtype=torch.long)


def evaluate_temperatures(tokens: torch.Tensor, vocab_size: int) -> List[MatchMetrics]:
    value = symbols_to_embeddings(tokens, vocab_size=vocab_size)
    hard_output, hard_aux = hard_rosa_reference(tokens, tokens, value)

    query = symbols_to_embeddings(tokens, vocab_size=vocab_size)
    key = query.clone()

    results: List[MatchMetrics] = []
    valid = hard_aux["best_j"] >= 0
    hard_matched_tokens = hard_output.argmax(dim=-1)

    for alpha, gamma in [(2.0, 2.0), (6.0, 6.0), (12.0, 12.0), (24.0, 24.0), (48.0, 48.0)]:
        soft_output, aux = soft_rosa_forward(
            query,
            key,
            value,
            alpha=alpha,
            gamma=gamma,
            similarity="cosine_margin",
            return_aux=True,
        )

        soft_best_j = aux["best_j"]
        soft_tokens = soft_output.argmax(dim=-1)

        position_acc = (soft_best_j[valid] == hard_aux["best_j"][valid]).float().mean().item()
        token_acc = (soft_tokens[valid] == hard_matched_tokens[valid]).float().mean().item()
        output_mse = F.mse_loss(soft_output[valid], hard_output[valid]).item()

        results.append(
            MatchMetrics(
                alpha=alpha,
                gamma=gamma,
                match_position_acc=position_acc,
                matched_token_acc=token_acc,
                output_mse=output_mse,
                matched_positions=int(valid.sum().item()),
            )
        )
    return results


def gradient_sanity_check() -> Tuple[float, float, float]:
    torch.manual_seed(0)
    query = torch.randn(2, 10, 4, requires_grad=True)
    key = torch.randn(2, 10, 4, requires_grad=True)
    value = torch.randn(2, 10, 6, requires_grad=True)

    output = soft_rosa_forward(
        query,
        key,
        value,
        alpha=7.0,
        gamma=4.0,
        similarity="cosine",
    )
    loss = output.square().mean()
    loss.backward()

    q_norm = query.grad.norm().item()
    k_norm = key.grad.norm().item()
    v_norm = value.grad.norm().item()
    return q_norm, k_norm, v_norm


def toy_training_demo(tokens: torch.Tensor, vocab_size: int) -> Tuple[float, float]:
    torch.manual_seed(0)
    device = tokens.device
    embed_dim = 8

    q_embed = nn.Embedding(vocab_size, embed_dim).to(device)
    k_embed = nn.Embedding(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.AdamW(list(q_embed.parameters()) + list(k_embed.parameters()), lr=0.15)

    target_value = symbols_to_embeddings(tokens, vocab_size=vocab_size).to(device)
    hard_output, hard_aux = hard_rosa_reference(tokens, tokens, target_value)
    valid = hard_aux["best_len"] > 0

    initial_loss = None
    for step in range(150):
        alpha = 2.0 + 18.0 * step / 149.0
        gamma = 2.0 + 18.0 * step / 149.0

        query = q_embed(tokens)
        key = k_embed(tokens)
        soft_output = soft_rosa_forward(
            query,
            key,
            target_value,
            alpha=alpha,
            gamma=gamma,
            similarity="cosine_margin",
        )
        loss = F.mse_loss(soft_output[valid], hard_output[valid])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = float(loss.item())

    final_loss = float(loss.item())
    final_output, aux = soft_rosa_forward(
        q_embed(tokens),
        k_embed(tokens),
        target_value,
        alpha=24.0,
        gamma=24.0,
        similarity="cosine_margin",
        return_aux=True,
    )
    token_acc = (
        final_output.argmax(dim=-1)[valid] == hard_output.argmax(dim=-1)[valid]
    ).float().mean().item()
    return float(initial_loss), final_loss, token_acc


def main() -> None:
    torch.manual_seed(0)
    tokens = build_toy_sequence()
    vocab_size = int(tokens.max().item()) + 1

    print("Soft ROSA demo")
    print(f"Toy tokens: {tokens.tolist()[0]}")
    print()

    print("[1] Approximation to Hard ROSA as alpha/gamma increase")
    for metrics in evaluate_temperatures(tokens, vocab_size):
        print(
            f"alpha={metrics.alpha:>5.1f} gamma={metrics.gamma:>5.1f} | "
            f"match-pos-acc={metrics.match_position_acc:.3f} | "
            f"matched-token-acc={metrics.matched_token_acc:.3f} | "
            f"output-mse={metrics.output_mse:.6f} | "
            f"matched-positions={metrics.matched_positions}"
        )
    print()

    print("[2] Gradient sanity check")
    q_norm, k_norm, v_norm = gradient_sanity_check()
    print(f"grad-norm(query)={q_norm:.6f}")
    print(f"grad-norm(key)  ={k_norm:.6f}")
    print(f"grad-norm(value)={v_norm:.6f}")
    print()

    print("[3] Tiny trainability demo")
    initial_loss, final_loss, token_acc = toy_training_demo(tokens, vocab_size)
    print(f"initial-loss={initial_loss:.6f}")
    print(f"final-loss  ={final_loss:.6f}")
    print(f"final-token-acc-on-matched-positions={token_acc:.3f}")


if __name__ == "__main__":
    main()
