# lmeval.py — lm-evaluation-harness 入口，适配 v18 架构与自动结果命名
import os
import sys
import json
import argparse
from datetime import datetime

import torch
from CaMoE.backbone import init_rwkv7_cuda
import lm_eval
from CaMoE.wrapper import CaMoELM


def main():
    parser = argparse.ArgumentParser(description="CaMoE lm-eval: 使用 get_config(scale) 与 checkpoint 内 config 匹配架构")
    parser.add_argument("--pretrained", type=str, default=None, help="Checkpoint 路径，例如 checkpoints/v18_0.4b/v18_step2000.pth")
    parser.add_argument("--scale", type=str, default="0.4b", choices=["0.1b", "0.4b"], help="未从 checkpoint 读 config 时使用的规模")
    parser.add_argument("--tasks", type=str, default="arc_easy", help="任务名，逗号分隔，如 arc_easy,hellaswag")
    parser.add_argument("--batch_size", type=int, default=64, help="评估 batch size")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 路径；不指定则自动生成 results_{version}_{scale}_{tasks}_{timestamp}.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    tasks_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks_list:
        tasks_list = ["arc_easy"]

    print("⏳ Init RWKV-7 CUDA Kernel...")
    init_rwkv7_cuda()
    print("🚀 Loading model...")
    lm = CaMoELM(
        pretrained=args.pretrained,
        scale=args.scale,
        device=args.device,
        batch_size=args.batch_size,
    )
    print("✅ Model ready!")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks_list,
        batch_size=args.batch_size,
    )

    print("\n📊 结果:")
    print(results.get("results", results))

    # 自动生成 JSON 文件名：results_{version}_{scale}_{tasks}_{timestamp}.json
    if args.output:
        out_path = args.output
    else:
        version = lm.config.get("version", "v18")
        scale = lm.config.get("scale", "0.4b")
        task_str = "_".join(tasks_list)[:64]  # 避免过长
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"results_{version}_{scale}_{task_str}_{ts}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved: {out_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
