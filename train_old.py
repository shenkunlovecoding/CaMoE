"""
CaMoE v10.7 训练脚本 (Final Fix)
适配: 本地Dataset / BF16 / Checkpointing / Gather-Scatter
"""

import os
import time
import argparse
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import load_from_disk
import bitsandbytes as bnb

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

from camoe import CaMoE_System
from config import CONFIG_01B, CONFIG_04B

try:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
except ImportError:
    TRIE_TOKENIZER = None


def get_phase(step: int, config: dict) -> str:
    # 简单的阶段调度
    if step < config.get('prewarm_steps', 100):
        return "prewarm"
    if step < config.get('warmup_steps', 500):
        return "warmup"
    return "normal"


def apply_phase(model, optimizer, phase: str, config: dict):
    # 动态调整学习率和冻结参数
    if phase == "prewarm":
        # 预热期：只训练新加的层 (Trans专家, Bridge)
        for n, p in model.named_parameters():
            should_train = any([
                'experts.' + str(model.num_rwkv_experts) in n,  # Trans专家
                'bridge' in n,
                'critic' in n,
                'capital' in n
            ])
            p.requires_grad = should_train
        lr = config.get('lr_prewarm', 1e-4)
    elif phase == "warmup":
        # 全量预热
        for p in model.parameters():
            p.requires_grad = True
        lr = config.get('lr_warmup', 2e-4)
    else:
        # 正常训练
        for p in model.parameters():
            p.requires_grad = True
        lr = config.get('lr_normal', 3e-4)
    
    # 更新优化器LR
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def load_weights(model, path):
    """从 RWKV 底模加载权重"""
    if not os.path.exists(path):
        print(f"⚠️ Weights not found: {path} (Starting from scratch)")
        return
    
    print(f"📦 Loading backbone from {path}...")
    official = torch.load(path, map_location='cpu', weights_only=True)
    my_dict = model.state_dict()
    loaded = 0
    
    for k, v in official.items():
        # 1. 直接匹配的层 (LN, Embedding, Head)
        if k in my_dict and my_dict[k].shape == v.shape:
            my_dict[k].copy_(v)
            loaded += 1
            continue
        
        # 2. Expert 映射 (把 RWKV Block 里的 FFN 权重复制给 RWKV 专家)
        # RWKV-6/7 Block 通常包含: att (TimeMix) 和 ffn (ChannelMix)
        if 'blocks' in k:
            try:
                # k 例子: blocks.0.ffn.key.weight
                parts = k.split('.')
                lid = int(parts[1])
                layer_type = parts[2] # att or ffn
                
                # Backbone (TimeMix) 直接加载
                if layer_type == 'att':
                    # 重新组装名字: blocks.0.att.xxx
                    target_name = f"blocks.{lid}.att.{'.'.join(parts[3:])}"
                    if target_name in my_dict and my_dict[target_name].shape == v.shape:
                        my_dict[target_name].copy_(v)
                        loaded += 1
                
                # FFN -> 复制给所有 RWKV Experts
                elif layer_type == 'ffn':
                    # parts[3] 可能是 key.weight, value.weight, receptance.weight
                    param_name = '.'.join(parts[3:])
                    
                    # 遍历所有 RWKV 专家
                    for i in range(model.num_rwkv_experts):
                        # 构造目标名字: blocks.0.experts.0.key.weight
                        # 注意：RWKV7 FFN 专家里可能叫 key/value，底模里可能叫 key/receptance
                        # 这里做一个简单的映射尝试
                        target = f"blocks.{lid}.experts.{i}.{param_name}"
                        
                        if target in my_dict and my_dict[target].shape == v.shape:
                            # 加上微小噪声，让专家初始状态略有不同
                            noise = torch.randn_like(v) * 0.01
                            my_dict[target].copy_(v + noise)
                            # 只计数一次，避免打印太多
                            if i == 0: loaded += 1
            except Exception as e:
                pass
    
    model.load_state_dict(my_dict, strict=False)
    print(f"✅ Loaded matching tensors (~{loaded})")


def log_gpu():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {alloc:.1f}/{total:.1f}GB"
    return ""


# 无限数据加载器
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="0.1b", choices=["0.1b", "0.4b"])
    args = parser.parse_args()
    
    config = CONFIG_01B if args.scale == "0.1b" else CONFIG_04B
    
    # 强制覆盖一些参数以适应显存
    config['num_rwkv_experts'] = 3 # 保持你的设置
    config['micro_batch_size'] = 6 # 如果显存够大，可以改大
    config['grad_accum'] = 8       # 梯度累积
    config['total_steps'] = 20000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    
    # 1. 加载 Tokenizer (仅用于 Logging 文本还原，不用于训练)
    if TRIE_TOKENIZER and os.path.exists(config['vocab_file']):
        tokenizer = TRIE_TOKENIZER(config['vocab_file'])
        print("✅ Tokenizer loaded")
    else:
        print("⚠️ Tokenizer not found (Logging will be silent)")
        tokenizer = None

    # 2. 加载数据集
    print("🚀 Loading pre-processed dataset from disk...")
    try:
        dataset = load_from_disk("./data/tinystories_processed")
        dataset.set_format(type="torch", columns=["input_ids"])
        print(f"📊 Dataset Size: {len(dataset)} sequences")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 3. DataLoader
    def simple_collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        # 截断到 config['ctx_len'] + 1 (因为要有 target)
        max_len = min(max_len, config['ctx_len'] + 1)
        
        CHUNK_LEN = 16
        # 计算输入部分需要的长度 (向上取整到 16 的倍数)
        input_len = ((max_len - 1 + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
        # 加上 target 的 1 位
        target_len = input_len + 1
        
        # 确保不短于最小长度 (至少要跑一个 chunk)
        target_len = max(target_len, CHUNK_LEN + 1)
        
        padded_batch = torch.zeros(len(batch), target_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            # 截断
            l = min(len(ids), target_len)
            padded_batch[i, :l] = ids[:l]
            
        return padded_batch
    train_loader = DataLoader(
        dataset, 
        batch_size=config['micro_batch_size'], 
        shuffle=True, 
        num_workers=0, 
        collate_fn=simple_collate,
        pin_memory=True
    )
    
    # 转换为无限迭代器
    train_iter = infinite_loader(train_loader)

    print("✅ DataLoader ready. Starting training loop...")
    
    # 4. 模型初始化
    model = CaMoE_System(config).to(device)
    
    # 开启显存优化
    model.gradient_checkpointing_enable()
    
    # 加载权重
    load_weights(model, config['weights_path'])
    
    print(f"📊 Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"   {log_gpu()}")
    
    # 优化器
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config['lr_prewarm'])
    
    if HAS_SWANLAB:
        swanlab.init(project=config['project'], name=config['run_name'], config=config)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"🚀 Training on {device} for {config['total_steps']} steps")
    
    # 5. 训练循环
    for step in range(config['total_steps']):
        t0 = time.time()
        
        # 阶段调度
        phase = get_phase(step, config)
        apply_phase(model, optimizer, phase, config)
        
        # 获取数据 (从 DataLoader)
        try:
            x_batch = next(train_iter)
        except StopIteration:
            # 理论上不会发生，因为是 infinite_loader
            print("⚠️ Data exhausted, restarting iterator...")
            train_iter = infinite_loader(train_loader)
            x_batch = next(train_iter)
            
        x_batch = x_batch.to(device)
        
        # 确保数据够长
        if x_batch.shape[1] <= 1:
            continue
            
        x, y = x_batch[:, :-1], x_batch[:, 1:]
        
        # Forward (混合精度)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, info = model(x, step=step, phase=phase)
            total_loss, token_losses, main_loss, critic_loss = model.compute_losses(logits, y, info)
            
            # 梯度累积平均
            loss_to_backward = total_loss / config['grad_accum']
        
        # Backward
        loss_to_backward.backward()
        
        # Optimizer Step
        if (step + 1) % config['grad_accum'] == 0:
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新市场 (仅在 Normal 阶段)
            if phase == "normal" and step > 100:
                model.update_market(info, token_losses, step)
        
        # Logging
        if step % 10 == 0: # 稍微频繁一点，方便看初期效果
            dt = time.time() - t0
            stats = model.log_market_health()
            
            # 计算 Tokens Per Second
            tps = config['micro_batch_size'] * x.shape[1] / dt
            
            # 拿到 Trans 专家的份额 (如果有的话)
            trans_share = stats.get("L0/TransShare", 0)
            if isinstance(trans_share, torch.Tensor): trans_share = trans_share.item()
            
            print(f"Step {step} | Loss: {main_loss.item():.3f} | "
                  f"Trans%: {trans_share:.2f} | TPS: {tps:.0f} | "
                  f"[{phase.upper()}] | {log_gpu()}")
            
            if HAS_SWANLAB:
                swanlab.log({
                    "Loss/Main": main_loss.item(),
                    "Loss/Critic": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
                    "Speed/TPS": tps,
                    **stats
                })
        
        # Save
        if step > 0 and step % 2000 == 0:
            path = os.path.join(config['save_dir'], f"v10_step{step}.pth")
            torch.save(model.state_dict(), path)
            print(f"💾 Saved: {path}")
    
    torch.save(model.state_dict(), os.path.join(config['save_dir'], "v10_final.pth"))
    print("🎉 Done!")


if __name__ == "__main__":
    main()