"""
CaMoE v12.0 训练脚本 (带 Eval Loss)
支持: 断点续训 / 自动步数识别 / 混合精度 / 显存优化 / 验证集评估
"""

import os
import gc
import time
import argparse
import re
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, DatasetDict
import bitsandbytes as bnb

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

from camoe import CaMoE_System
from config import *

try:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
except ImportError:
    TRIE_TOKENIZER = None

def load_backbone(model, path):
    """从 RWKV 底模加载权重"""
    if not os.path.exists(path):
        print(f"⚠️ Weights not found: {path} (Starting from scratch)")
        return
    
    print(f"📦 Loading backbone from {path}...")
    official = torch.load(path, map_location='cpu', weights_only=True)
    my_dict = model.state_dict()
    loaded = 0
    
    for k, v in official.items():
        if k in my_dict and my_dict[k].shape == v.shape:
            my_dict[k].copy_(v)
            loaded += 1
            continue
        
        if 'blocks' in k:
            try:
                parts = k.split('.')
                lid = int(parts[1])
                layer_type = parts[2]
                
                if layer_type == 'att':
                    target_name = f"blocks.{lid}.att.{'.'.join(parts[3:])}"
                    if target_name in my_dict and my_dict[target_name].shape == v.shape:
                        my_dict[target_name].copy_(v)
                        loaded += 1
                
                elif layer_type == 'ffn':
                    param_name = '.'.join(parts[3:])
                    for i in range(model.num_rwkv_experts):
                        target = f"blocks.{lid}.experts.{i}.{param_name}"
                        if target in my_dict and my_dict[target].shape == v.shape:
                            noise = torch.randn_like(v) * 0.01
                            my_dict[target].copy_(v + noise)
                            if i == 0: loaded += 1
            except Exception as e:
                pass
    
    model.load_state_dict(my_dict, strict=False)
    print(f"✅ Loaded matching tensors (~{loaded})")

def get_phase(step: int, config: dict) -> str:
    if step < config.get('prewarm_steps', 100):
        return "prewarm"
    if step < config.get('warmup_steps', 500):
        return "warmup"
    return "normal"

def apply_phase(model, optimizer, phase: str, config: dict):
    num_rwkv = config.get('num_rwkv_experts', 2)
    num_trans = config.get('num_trans_experts', 1)
    
    if phase == "prewarm":
        trans_indices = [str(i) for i in range(num_rwkv, num_rwkv + num_trans)]
        for n, p in model.named_parameters():
            is_trans_expert = any(f'experts.{idx}.' in n for idx in trans_indices)
            should_train = any([is_trans_expert, 'bridge' in n, 'critic' in n, 'capital' in n])
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
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config = CONFIG_BABYLM
    
    # 强制设置 Eval 频率
    eval_interval = config.get('eval_interval', 500)  # 每500步评测一次
    eval_iters = config.get('eval_iters', 50)         # 每次评测跑50个batch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    # 1. Tokenizer
    if TRIE_TOKENIZER and os.path.exists(config['vocab_file']):
        tokenizer = TRIE_TOKENIZER(config['vocab_file'])
    else:
        tokenizer = None

    # 2. Dataset & Split
    print("🚀 Loading dataset...")
    try:
        raw_dataset = load_from_disk(config.get('data_path'))
        
        # [修改] 自动划分 训练集/验证集
        if isinstance(raw_dataset, DatasetDict):
            if 'validation' in raw_dataset:
                train_data = raw_dataset['train']
                val_data = raw_dataset['validation']
            elif 'test' in raw_dataset:
                train_data = raw_dataset['train']
                val_data = raw_dataset['test']
            else:
                # 只有 train，手动切分
                split = raw_dataset['train'].train_test_split(test_size=0.05, seed=42)
                train_data = split['train']
                val_data = split['test']
        elif isinstance(raw_dataset, Dataset):
            # 单个 Dataset，手动切分
            split = raw_dataset.train_test_split(test_size=0.05, seed=42)
            train_data = split['train']
            val_data = split['test']
        else:
            raise ValueError("Unknown dataset type")

        train_data.set_format(type="torch", columns=["input_ids"])
        val_data.set_format(type="torch", columns=["input_ids"])
        
        print(f"📊 Dataset Split: Train={len(train_data)}, Val={len(val_data)}")

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
        train_data, batch_size=config['micro_batch_size'], shuffle=True, 
        num_workers=0, collate_fn=simple_collate, pin_memory=True
    )
    # [新增] 验证集 Loader
    val_loader = DataLoader(
        val_data, batch_size=config['micro_batch_size'], shuffle=False, 
        num_workers=0, collate_fn=simple_collate, pin_memory=True
    )
    
    train_iter = infinite_loader(train_loader)

    # 4. Model & Optimizer
    print("🏗️ Building model...")
    model = CaMoE_System(config).to(device)
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config['lr_prewarm'])

    # ==========================================
    # 断点续训逻辑
    # ==========================================
    start_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"🔄 Resuming from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_step = checkpoint['step'] + 1
                print(f"✅ Full state restored. Resuming from step {start_step}")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("⚠️ Loaded weights only.")
                match = re.search(r'step(\d+)', args.resume)
                if match:
                    start_step = int(match.group(1)) + 1
        else:
            print(f"❌ Checkpoint {args.resume} not found!")
            return
    else:
        load_backbone(model, config['weights_path'])
    
    # ==========================================
    # [新增] 评估函数
    # ==========================================
    @torch.no_grad()
    def estimate_loss(model, loader, eval_steps):
        model.eval()
        losses = []
        val_iter = iter(loader)
        
        for _ in range(eval_steps):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(loader)
                batch = next(val_iter)
            
            batch = batch.to(device)
            if batch.shape[1] <= 1: continue
            
            x, y = batch[:, :-1], batch[:, 1:]
            
            # Eval 时使用 Normal 模式，测试全系统
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, info = model(x, step=100000, phase="normal") # 这里的 step 传大一点确保触发 market
                # 只算 Main Loss
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            
            losses.append(loss.item())
        
        model.train()
        return sum(losses) / len(losses)

    print(f"📊 Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    if HAS_SWANLAB:
        swanlab.init(project=config['project'], name=config['run_name'], config=config)
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"🚀 Training start from step {start_step}...")
    
    # 5. Training Loop
    for step in range(start_step, config['total_steps']):
        torch.cuda.reset_peak_memory_stats()
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
            total_loss, token_losses, main_loss, critic_loss ,bridge_loss = model.compute_losses(logits, y, info)
            loss_to_backward = total_loss / config['grad_accum']

        loss_to_backward.backward()
        
        if (step + 1) % config['grad_accum'] == 0:
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if phase == "normal" and step > 100:
                model.update_market(info, token_losses, step)
        
        # [修改] 日志与评估逻辑
        if step % 10 == 0:
            dt = time.time() - t0
            tps = config['micro_batch_size'] * x.shape[1] / dt
            
            # --- 评估 ---
            val_loss = None
            if step > 0 and step % eval_interval == 0:
                print(f"🔍 Evaluating at step {step}...")
                val_loss = estimate_loss(model, val_loader, eval_iters)
            
            # 统计
            stats = model.log_market_health()
            trans_share = stats.get("L0/TransShare", 0)
            if isinstance(trans_share, torch.Tensor): trans_share = trans_share.item()
            
            # 打印
            log_str = f"Step {step} | TrainLoss: {main_loss.item():.3f}"
            if val_loss:
                log_str += f" | ValLoss: {val_loss:.3f}"
            log_str += f" | Trans%: {trans_share:.1f} | TPS: {tps:.0f} | [{phase.upper()}]"
            print(log_str)
            
            # SwanLab
            if HAS_SWANLAB:
                logs = {
                    "Loss/Train_Main": main_loss.item(),
                    "Loss/Train_Critic": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
                    "Loss/Train_Bridge" : bridge_loss.item() if isinstance(bridge_loss, torch.Tensor) else bridge_loss,
                    "Speed/TPS": tps,
                    **stats
                }
                if val_loss:
                    logs["Loss/Validation"] = val_loss
                swanlab.log(logs)
        
        # 保存完整 Checkpoint
        if step > 0 and step % 2000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 Cache cleared")
            path = os.path.join(config['save_dir'], f"v12_step{step}.pth")
            
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': config
            }
            torch.save(checkpoint, path)
            print(f"💾 Saved Checkpoint: {path}")
    
    final_path = os.path.join(config['save_dir'], "v12_final.pth")
    torch.save({'model': model.state_dict(), 'step': config['total_steps']}, final_path)
    print("🎉 Done!")

if __name__ == "__main__":
    main()