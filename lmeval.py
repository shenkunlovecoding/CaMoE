# lmeval.py
import os
import sys
print("Step 1: 环境变量设置完成")

import torch
print(f"Step 2: PyTorch 导入完成, CUDA={torch.cuda.is_available()}")

print("Step 3: 准备导入 wrapper...")

# 分步导入，看卡在哪
print("  3.1: 导入 camoe...")
from camoe import CaMoE_System
print("  3.2: 导入 backbone...")
from camoe.backbone import init_rwkv7_cuda
print("  3.3: 初始化 CUDA kernel (可能要几分钟)...")
init_rwkv7_cuda()
print("  3.4: 导入 config...")
from camoe.config import CONFIG_MINIPILE
print("  3.5: 导入 tokenizer...")
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

print("Step 4: 所有导入完成，开始构建模型...")
import json
import lm_eval
from camoe.wrapper import CaMoELM

def main():
    print("🚀 主进程启动，开始加载模型...")
    
    lm = CaMoELM(
        pretrained="checkpoints/minipile/v12_final.pth",
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