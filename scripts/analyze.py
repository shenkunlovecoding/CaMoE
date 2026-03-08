"""Analyze expert-capital checkpoints for CaMoE v22."""

from __future__ import annotations

import argparse
import glob
import os

import torch


def analyze_checkpoint(path: str) -> dict | None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    step = checkpoint.get("step", "?") if isinstance(checkpoint, dict) else "?"
    capitals = state_dict.get("capital_manager.capitals")
    if capitals is None:
        return {"path": os.path.basename(path), "step": step, "error": "capital_manager.capitals missing"}

    layers = []
    for layer_idx in range(capitals.size(0)):
        caps = capitals[layer_idx].float()
        shares = caps / caps.sum().clamp(min=1e-6)
        sorted_caps, _ = torch.sort(caps)
        n = sorted_caps.numel()
        idx = torch.arange(1, n + 1, dtype=sorted_caps.dtype)
        gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * caps.sum().clamp(min=1e-6))
        layers.append(
            {
                "layer": layer_idx,
                "gini": float(gini.item()),
                "winner": int(caps.argmax().item()),
                "shares": [float(value) for value in shares.tolist()],
            }
        )

    return {
        "path": os.path.basename(path),
        "step": step,
        "layers": layers,
        "avg_gini": sum(layer["gini"] for layer in layers) / max(len(layers), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CaMoE v22 checkpoints")
    parser.add_argument("--dir", default="checkpoints")
    parser.add_argument("--pattern", default="*.pth")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("No checkpoints found.")
        return

    for path in paths:
        result = analyze_checkpoint(path)
        if result is None:
            print(f"{os.path.basename(path)} failed_to_load")
            continue
        if "error" in result:
            print(f"{result['path']} step={result['step']} error={result['error']}")
            continue
        print(f"{result['path']} step={result['step']} avg_gini={result['avg_gini']:.4f}")
        for layer in result["layers"]:
            share_str = " ".join(f"E{i}:{share:.3f}" for i, share in enumerate(layer["shares"]))
            print(f"  L{layer['layer']:02d} gini={layer['gini']:.4f} winner=E{layer['winner']} {share_str}")


if __name__ == "__main__":
    main()
