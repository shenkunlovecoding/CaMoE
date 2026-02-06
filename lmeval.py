# lmeval.py
import os
import sys
print("Step 1: 环境变量设置完成")

import torch
print(f"Step 2: PyTorch 导入完成, CUDA={torch.cuda.is_available()}")
print("  3.2: 导入 backbone...")
from CaMoE.backbone import init_rwkv7_cuda
init_rwkv7_cuda()
import json
import lm_eval
from CaMoE.wrapper import CaMoELM

def main():
    print("🚀 主进程启动，开始加载模型...")
    
    lm = CaMoELM(
        pretrained="checkpoints/minipile/v16_step12000.pth",
        device="cuda",
        batch_size=1,
    )
    print("✅ Model ready!")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["arc_easy"],
        batch_size=64
    )

    print("\n📊 结果:")
    print(results["results"])
    with open("results_sst2_38k.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("💾 Results cached")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()