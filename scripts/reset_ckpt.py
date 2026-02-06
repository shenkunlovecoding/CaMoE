import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="checkpoints/babylm/v12_step38000.pth", help="Input checkpoint path")
    parser.add_argument("--output", "-o", type=str, default="checkpoints/minipile/v12_minipile_init.pth", help="Output checkpoint path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file {args.input} not found!")
        return

    print(f"📦 Loading checkpoint from {args.input}...")
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)

    # 1. 提取模型权重
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model_state = ckpt['model']
    else:
        model_state = ckpt
        
    print(f"✅ Model weights loaded. Keys: {len(model_state)}")

    # 2. 检查市场权重 (CapitalManager)
    # 我们希望保留 Gini 和 资本分配，这部分就在 model_state 里 (buffer)
    if "capital_manager.capitals" in model_state:
        print("💰 Capital Manager state found! Preserving market economy...")
        print(f"   Capitals shape: {model_state['capital_manager.capitals'].shape}")
    else:
        print("⚠️ Warning: Capital Manager state NOT found. Market will reset to communism.")

    # 3. 创建新的干净 Checkpoint
    # 丢弃 'optimizer', 'step', 'config' (因为我们要用新的 config)
    new_ckpt = {
        'model': model_state,
        # 'optimizer': ... 丢弃，让 train.py 重新初始化
        'step': 0, # 重置步数
        'info': "Reset for MiniPile Curriculum Learning"
    }

    # 4. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(new_ckpt, args.output)
    
    print("-" * 40)
    print(f"🎉 Success! Clean checkpoint saved to: {args.output}")
    print(f"🚀 You can now start training on MiniPile from step 0.")
    print("-" * 40)

if __name__ == "__main__":
    main()