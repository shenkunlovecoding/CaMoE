"""
CaMoE v11.0 训练脚本
支持: 断点续训 / 自动步数识别 / 混合精度 / 显存优化
"""

import os
import time
import argparse
import re  # 用于解析文件名里的步数
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
    if step < config.get('prewarm_steps', 100):
        return "prewarm"
    if step < config.get('warmup_steps', 500):
        return "warmup"
    return "normal"


def apply_phase(model, optimizer, phase: str, config: dict):
    # ... (保持原样) ...
    if phase == "prewarm":
        for n, p in model.named_parameters():
            should_train = any([
                'experts.' + str(model.num_rwkv_experts) in n,
                'bridge' in n,
                'critic' in n,
                'capital' in n
            ])
            p.requires_grad = should_train
        lr = config.get('lr_prewarm', 1e-4)
    elif phase == "warmup":
        for p in model.parameters():
            p.requires_grad = True
        lr = config.get('lr_warmup', 2e-4)
    else:
        for p in model.parameters():
            p.requires_grad = True
        lr = config.get('lr_normal', 3e-4)
    
    for pg in optimizer.param_groups:
        pg['lr'] = lr


from train_old import load_weights as load_backbone


def log_gpu():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {alloc:.1f}/{total:.1f}GB"
    return ""


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="0.1b", choices=["0.1b", "0.4b"])
    # [新增] Resume 参数
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config = CONFIG_01B if args.scale == "0.1b" else CONFIG_04B
    
    # 强制覆盖
    config['num_rwkv_experts'] = 3
    config['micro_batch_size'] = 6
    config['grad_accum'] = 8
    config['total_steps'] = 20000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')
    
    # 1. Tokenizer
    if TRIE_TOKENIZER and os.path.exists(config['vocab_file']):
        tokenizer = TRIE_TOKENIZER(config['vocab_file'])
    else:
        tokenizer = None

    # 2. Dataset
    print("🚀 Loading dataset...")
    try:
        dataset = load_from_disk("./data/tinystories_processed")
        dataset.set_format(type="torch", columns=["input_ids"])
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 3. DataLoader & Collate
    def simple_collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, config['ctx_len'] + 1)
        
        # [CUDA Kernel 要求] 对齐到 16 的倍数 + 1
        CHUNK_LEN = 16
        input_len = ((max_len - 1 + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
        target_len = max(input_len + 1, CHUNK_LEN + 1)
        
        padded_batch = torch.zeros(len(batch), target_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            l = min(len(ids), target_len)
            padded_batch[i, :l] = ids[:l]
        return padded_batch

    train_loader = DataLoader(
        dataset, batch_size=config['micro_batch_size'], shuffle=True, 
        num_workers=0, collate_fn=simple_collate, pin_memory=True
    )
    train_iter = infinite_loader(train_loader)

    # 4. Model & Optimizer
    print("🏗️ Building model...")
    model = CaMoE_System(config).to(device)
    #model.gradient_checkpointing_enable() Has Problem
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config['lr_prewarm'])

    # ==========================================
    # [核心修改] 断点续训逻辑
    # ==========================================
    start_step = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"🔄 Resuming from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # 判断是新版存档(dict)还是旧版存档(weights only)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # [新版] 完美恢复
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_step = checkpoint['step'] + 1
                print(f"✅ Full state restored. Resuming from step {start_step}")
            else:
                # [旧版] 只恢复权重，尝试从文件名猜步数
                model.load_state_dict(checkpoint, strict=False)
                print("⚠️ Loaded weights only (Old format). Optimizer reset.")
                
                # 尝试从文件名解析步数 (例如 v10_step1500.pth)
                match = re.search(r'step(\d+)', args.resume)
                if match:
                    start_step = int(match.group(1)) + 1
                    print(f"📅 Guessed start step: {start_step}")
                else:
                    print("⚠️ Could not guess step from filename, starting from 0 (but with trained weights)")
        else:
            print(f"❌ Checkpoint {args.resume} not found!")
            return
    else:
        # 没传 resume，尝试加载底模
        load_backbone(model, config['weights_path'])
    
    # ==========================================

    print(f"📊 Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    if HAS_SWANLAB:
        swanlab.init(project=config['project'], name=config['run_name'], config=config)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"🚀 Training start from step {start_step}...")
    
    # 5. Training Loop
    # 注意：这里 range 从 start_step 开始
    for step in range(start_step, config['total_steps']):
        t0 = time.time()
        
        phase = get_phase(step, config)
        apply_phase(model, optimizer, phase, config)
        
        try:
            x_batch = next(train_iter)
        except StopIteration:
            train_iter = infinite_loader(train_loader)
            x_batch = next(train_iter)
            
        x_batch = x_batch.to(device)
        if x_batch.shape[1] <= 1: continue
            
        x, y = x_batch[:, :-1], x_batch[:, 1:]
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, info = model(x, step=step, phase=phase)
            total_loss, token_losses, main_loss, critic_loss = model.compute_losses(logits, y, info)
            loss_to_backward = total_loss / config['grad_accum']
        
        loss_to_backward.backward()
        
        if (step + 1) % config['grad_accum'] == 0:
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if phase == "normal" and step > 100:
                model.update_market(info, token_losses, step)
        
        if step % 10 == 0:
            dt = time.time() - t0
            stats = model.log_market_health()
            tps = config['micro_batch_size'] * x.shape[1] / dt
            trans_share = stats.get("L0/TransShare", 0)
            if isinstance(trans_share, torch.Tensor): trans_share = trans_share.item()
            
            print(f"Step {step} | Loss: {main_loss.item():.3f} | "
                  f"Trans%: {trans_share:.1f} | TPS: {tps:.0f} | "
                  f"[{phase.upper()}] | {log_gpu()}")
            
            if HAS_SWANLAB:
                swanlab.log({
                    "Loss/Main": main_loss.item(),
                    "Loss/Critic": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
                    "Speed/TPS": tps,
                    **stats
                })
        
        # [修改] 保存完整 Checkpoint
        if step > 0 and step % 1000 == 0:
            path = os.path.join(config['save_dir'], f"v10_step{step}.pth")
            
            # 保存完整状态
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': config
            }
            torch.save(checkpoint, path)
            print(f"💾 Saved Checkpoint: {path}")
    
    # Final save
    final_path = os.path.join(config['save_dir'], "v10_final.pth")
    torch.save({'model': model.state_dict(), 'step': config['total_steps']}, final_path)
    print("🎉 Done!")

if __name__ == "__main__":
    main()