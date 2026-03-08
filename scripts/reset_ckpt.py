import torch
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Reset a CaMoE v22 checkpoint for fresh training.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output checkpoint path")
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

    # 2. 检查市场状态 buffer
    if "capital_manager.capitals" in model_state:
        print("💰 Capital state found and will be preserved.")
        print(f"   Capitals shape: {model_state['capital_manager.capitals'].shape}")
    else:
        print("⚠️ Warning: capital_manager.capitals not found. Expert capital will reset on next init.")

    # 3. 创建新的干净 checkpoint
    new_ckpt = {
        'model': model_state,
        'step': 0,
        'info': "Reset for CaMoE v22 fresh training",
    }
    if isinstance(ckpt, dict) and 'config' in ckpt:
        new_ckpt['config'] = ckpt['config']

    # 4. 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(new_ckpt, args.output)
    
    print("-" * 40)
    print(f"🎉 Success! Clean checkpoint saved to: {args.output}")
    print("🚀 You can now resume training from step 0 with freshly initialized optimizers.")
    print("-" * 40)

if __name__ == "__main__":
    main()
