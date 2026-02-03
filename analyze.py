import torch
import torch.nn as nn
from camoe import CaMoE_System

# ==========================================
# 1. 你的配置 (手动填入你 train.py 里用的 config)
# ==========================================
# 我根据 0.1B RWKV 的标准配置填了一个默认的，
# 如果你改过 train.py 里的 config，请在这里修改！
config = {
    'n_embd': 768,       # 0.1B 标准是 768
    'n_layer': 12,       # 0.1B 标准是 12
    'head_size': 64,
    'vocab_size': 65536, # 或者 50277
    
    # 关键嫌疑人：专家数量
    'num_rwkv_experts': 3, # 如果这里很大，参数会爆炸
    
    # 其他
    'ctx_len': 1024,
    'total_capital': 10000.0,
}

def analyze():
    print(f"🔍 正在分析 CaMoE 模型配置...")
    print(f"📋 Config: {config}")
    
    # 实例化模型 (不加载权重，只看骨架)
    model = CaMoE_System(config)
    
    # 1. 统计总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n" + "="*40)
    print(f"📊 参数量统计")
    print(f"="*40)
    print(f"Total Params:     {total_params / 1e6:.2f} M ({total_params / 1e9:.3f} B)")
    print(f"Trainable Params: {trainable_params / 1e6:.2f} M")
    
    # 2. 显存估算 (静态)
    # BF16 = 2 bytes
    model_mem_gb = total_params * 2 / (1024**3)
    # AdamW 优化器状态 (m, v) = 8 bytes (FP32) 或者 2 bytes (8-bit)
    optim_mem_8bit = total_params * 2 / (1024**3) # state + weight copy
    grad_mem = total_params * 2 / (1024**3) # gradients (BF16)
    
    print(f"\n💾 静态显存需求估算 (不含激活值)")
    print(f"--------------------------------")
    print(f"Model Weights (BF16): {model_mem_gb:.2f} GB")
    print(f"Gradients     (BF16): {grad_mem:.2f} GB")
    print(f"Optimizer (8-bit):    {optim_mem_8bit:.2f} GB")
    print(f"--------------------------------")
    print(f"🔥 仅启动就需要:      {model_mem_gb + grad_mem + optim_mem_8bit:.2f} GB")
    print(f"   (如果这是 1.7B 模型，启动就要 10GB+，还没开始跑数据)")

    # 3. 参数分布分析 (谁是胖子？)
    print(f"\n🥩 参数分布解剖")
    print(f"--------------------------------")
    
    backbone_params = 0
    experts_params = 0
    bridge_params = 0
    
    for name, module in model.named_modules():
        # 统计 Block 里的具体分布
        if isinstance(module, nn.ModuleList) and name == 'blocks':
            first_block = module[0]
            
            # 统计 Attn (Backbone)
            attn_p = sum(p.numel() for p in first_block.att.parameters())
            print(f"Layer 0 - RWKV TimeMix: {attn_p/1e6:.2f} M")
            
            # 统计 Experts
            exp_p_total = 0
            for i, exp in enumerate(first_block.experts):
                this_exp_p = sum(p.numel() for p in exp.parameters())
                if i < len(first_block.experts) - 1:
                    print(f"   ├─ RWKV Expert {i}:  {this_exp_p/1e6:.2f} M")
                else:
                    print(f"   └─ Trans Expert:    {this_exp_p/1e6:.2f} M")
                exp_p_total += this_exp_p
            
            print(f"Layer 0 - Total Experts: {exp_p_total/1e6:.2f} M")
            
            # 统计 Bridge
            bridge_p = sum(p.numel() for p in first_block.bridge.parameters())
            print(f"Layer 0 - Bridge:        {bridge_p/1e6:.2f} M")
            
            # 宏观推算
            total_experts_all_layers = exp_p_total * config['n_layer']
            print(f"\n👉 结论：全模型 Expert 参数总和 ≈ {total_experts_all_layers/1e6:.2f} M")
            if total_experts_all_layers > total_params * 0.8:
                print("⚠️ 警告：绝大部分参数都在专家层！")
                print("   MoE 极大地膨胀了显存需求，虽然计算量(FLOPs)没变，但显存必须存下所有专家。")
            break

    # 4. 模拟一次前向传播 (检查是否会瞬间 OOM)
    print(f"\n🧪 正在尝试 Dummy Forward (检查中间激活)...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        # 模拟一个 Batch
        x = torch.randint(0, config['vocab_size'], (4, config['ctx_len'])).to(device) # Batch=4
        
        torch.cuda.reset_peak_memory_stats()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, info = model(x)
            
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"✅ Forward 成功！")
        print(f"📈 峰值显存 (Batch=4, ctx={config['ctx_len']}): {peak_mem:.2f} GB")
        
        # 检查有没有 Broadcasting 炸裂
        print(f"   如果这里没报错，说明 [Batch, Batch] 的 Bug 修好了。")
        
    except Exception as e:
        print(f"❌ Forward 失败: {e}")
        print("   可能是显存不足，或者维度不匹配。")

if __name__ == "__main__":
    analyze()