"""Quick VRAM smoke profiler for CaMoE v22."""

from __future__ import annotations

import torch

from camoe.backbone import init_rwkv7_cuda
from camoe.config import get_config
from camoe.model import CaMoE_Model


def profile_vram(scale: str = "0.4b") -> None:
    config = get_config(scale)
    device = "cuda"
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    init_rwkv7_cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / 1024**2

    model = CaMoE_Model(config).to(device)
    model_mem = torch.cuda.memory_allocated() / 1024**2 - base_mem

    input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)
    targets = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len), device=device)
    result = model(input_ids, targets, critic_alpha=1.0, training=True, uniform=False)
    result["loss_scalar"].backward()

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"scale={scale} params_vram_mb={model_mem:.2f} peak_vram_mb={peak_mem:.2f} "
        f"loss={float(result['loss_scalar'].detach().item()):.4f}"
    )


if __name__ == "__main__":
    profile_vram("0.4b")
