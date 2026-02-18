import argparse
import os
import tempfile
import shutil
from typing import Dict, Tuple

import torch


def parse_args() -> argparse.Namespace:
    r"""parse_args() -> argparse.Namespace

    解析命令行参数。

    Returns:
      argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser(description="Slim CaMoE checkpoint for inference.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 checkpoint 路径（训练产物或纯 state_dict）。",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出推理 pth 路径。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["keep", "bf16", "fp16", "fp32"],
        help="浮点权重导出精度（默认 bf16，可显著减小体积）。",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="state_dict",
        choices=["state_dict", "checkpoint"],
        help="导出格式：state_dict(更小) 或 checkpoint({'model': ...})。",
    )
    parser.add_argument(
        "--strip-prefix",
        type=str,
        nargs="*",
        default=[],
        help="可选：移除指定前缀的权重键（谨慎使用）。示例 --strip-prefix optimizer",
    )
    return parser.parse_args()


def _target_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    return None


def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: expected dict or {'model': state_dict}.")


def _slim_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype_name: str,
    strip_prefix: Tuple[str, ...],
) -> Dict[str, torch.Tensor]:
    target_dtype = _target_dtype(dtype_name)
    out: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        if any(k.startswith(p) for p in strip_prefix):
            continue

        if torch.is_tensor(v):
            t = v.detach().cpu()
            if target_dtype is not None and torch.is_floating_point(t):
                t = t.to(target_dtype)
            out[k] = t.contiguous()
        else:
            out[k] = v

    return out


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input not found: {args.input}")
        return

    print(f"📦 Loading: {args.input}")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    print(f"✅ Loaded state_dict keys: {len(state_dict)}")

    slim_state = _slim_state_dict(
        state_dict=state_dict,
        dtype_name=args.dtype,
        strip_prefix=tuple(args.strip_prefix),
    )
    print(f"🧹 Slimmed keys: {len(slim_state)} | dtype={args.dtype}")

    if args.format == "checkpoint":
        export_obj = {"model": slim_state, "info": "Slim checkpoint for inference"}
    else:
        export_obj = slim_state

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 先保存到与目标同目录的临时文件，避免跨设备替换失败
    temp_dir = out_dir if out_dir else "."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth", dir=temp_dir) as f:
        tmp_path = f.name

    torch.save(export_obj, tmp_path)
    try:
        os.replace(tmp_path, args.output)
    except OSError:
        # 兜底：某些环境 replace 仍可能失败，退化为 move（可跨设备）
        shutil.move(tmp_path, args.output)

    src_mb = _file_size_mb(args.input)
    dst_mb = _file_size_mb(args.output)
    ratio = (dst_mb / src_mb) if src_mb > 0 else 0.0

    print("-" * 50)
    print(f"✅ Exported inference pth: {args.output}")
    print(f"📏 Size: {src_mb:.2f} MB -> {dst_mb:.2f} MB (x{ratio:.3f})")
    print("💡 建议 eval/infer 直接加载该文件作为 state_dict。")
    print("-" * 50)


if __name__ == "__main__":
    main()
