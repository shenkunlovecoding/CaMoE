"""Simple generation utility for CaMoE v22."""

from __future__ import annotations

import argparse
import os
from collections import Counter

import torch
import torch.nn.functional as F

from camoe.backbone import init_rwkv7_cuda
from camoe.config import get_config
from camoe.model import load_camoe_checkpoint

try:
    import pyrwkv_tokenizer

    RUST_TOKENIZER = True
except ImportError:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

    RUST_TOKENIZER = False


def load_tokenizer(vocab_file: str):
    if RUST_TOKENIZER:
        return pyrwkv_tokenizer.RWKVTokenizer()
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"Tokenizer vocab not found: {vocab_file}")
    return TRIE_TOKENIZER(vocab_file)


def pad_to_chunk(input_ids: torch.Tensor, chunk_len: int = 16) -> tuple[torch.Tensor, int]:
    batch, steps = input_ids.shape
    del batch
    if steps % chunk_len == 0:
        return input_ids, 0
    pad_len = chunk_len - (steps % chunk_len)
    padding = torch.zeros((input_ids.size(0), pad_len), dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, padding], dim=1), pad_len


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = F.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs).scatter(1, sorted_indices, sorted_probs * keep)
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.multinomial(filtered, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with CaMoE v22")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--scale", default="0.4b", choices=["0.1b", "0.4b"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--vocab_file", default="tokenizer/rwkv_vocab_v20230424.txt")
    args = parser.parse_args()

    init_rwkv7_cuda()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, config, _ = load_camoe_checkpoint(
        args.checkpoint,
        device=device,
        config=get_config(args.scale),
        strict=True,
    )
    model.eval()
    tokenizer = load_tokenizer(args.vocab_file)

    if RUST_TOKENIZER:
        token_ids = tokenizer.encode(args.prompt)
    else:
        token_ids = tokenizer.encode(args.prompt)

    sequence = torch.tensor([token_ids], dtype=torch.long, device=device)
    usage = [Counter() for _ in range(config.n_layers)]

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = sequence[:, -config.seq_len :]
            padded, pad_len = pad_to_chunk(context)
            result = model(
                padded,
                critic_alpha=1.0,
                training=False,
                uniform=False,
            )
            logits = result["logits"]
            if pad_len > 0:
                logits = logits[:, :-pad_len, :]
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, args.temperature, args.top_p)

            for layer_idx, block in enumerate(model.blocks):
                cache = block.get_cache()
                if not cache:
                    continue
                winners = cache["winners"][0, -1]
                usage[layer_idx].update(int(index) for index in winners.tolist())

            sequence = torch.cat([sequence, next_token], dim=1)
            if int(next_token.item()) == 0:
                break

    generated_ids = sequence[0].tolist()
    if RUST_TOKENIZER:
        text = tokenizer.decode(generated_ids)
    else:
        text = tokenizer.decode(generated_ids)

    print(text)
    print("\nPer-layer winner counts:")
    for layer_idx, counter in enumerate(usage):
        if not counter:
            continue
        summary = " ".join(f"E{expert}:{count}" for expert, count in counter.most_common())
        print(f"L{layer_idx:02d} {summary}")


if __name__ == "__main__":
    main()
