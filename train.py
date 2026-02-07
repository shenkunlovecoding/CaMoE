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
from datasets import load_from_disk, Dataset, DatasetDict, interleave_datasets
import bitsandbytes as bnb
from CaMoE.backbone import init_rwkv7_cuda
try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

from CaMoE.system import CaMoE_System
from CaMoE.config import get_config, VERSION


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
    init_rwkv7_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="0.4b", choices=["0.1b", "0.4b"])
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config = get_config(args.scale)
    
    # 强制设置 Eval 频率
    eval_interval = config.get('eval_interval', 1000)  # 每500步评测一次
    eval_iters = config.get('eval_iters', 50)         # 每次评测跑50个batch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    # ==========================================
    # 2. Dataset & Split（支持多数据集混合，手动 Resume 换阶段）
    # ==========================================
    print("🚀 Loading datasets...")
    try:
        mix = config.get("mix")
        data_roots = config.get("data_roots") or {}

        if mix and data_roots:
            # 混合模式：按 mix 比例 interleave，课程学习时改 config + Resume 即可
            train_datasets = []
            val_datasets = []
            probs = []
            loaded_names = []

            for name, prob in mix.items():
                if prob <= 0:
                    continue
                path = data_roots.get(name)
                if not path or not os.path.exists(path):
                    print(f"⚠️ Dataset not found: {path}, skipping {name}.")
                    continue

                ds = load_from_disk(path)
                if isinstance(ds, DatasetDict):
                    tr = ds["train"]
                    va = ds.get("validation") or ds.get("test")
                    if va is None:
                        split = tr.train_test_split(test_size=0.01, seed=42)
                        tr, va = split["train"], split["test"]
                else:
                    split = ds.train_test_split(test_size=0.01, seed=42)
                    tr, va = split["train"], split["test"]

                tr.set_format(type="torch", columns=["input_ids"])
                va.set_format(type="torch", columns=["input_ids"])
                train_datasets.append(tr)
                val_datasets.append(va)
                probs.append(prob)
                loaded_names.append(name)
                print(f"  - {name}: train={len(tr)}, val={len(va)} (prob={prob})")

            if not train_datasets:
                raise ValueError("No valid datasets in mix (paths missing or prob=0).")

            total_p = sum(probs)
            probs = [p / total_p for p in probs]
            train_data = interleave_datasets(train_datasets, probabilities=probs, seed=42,stopping_strategy="all_exhausted")
            val_data = interleave_datasets(val_datasets, probabilities=probs, seed=42,stopping_strategy="all_exhausted")
            print(f"📊 Mix: {dict(zip(loaded_names, probs))} → Train={len(train_data)}, Val={len(val_data)}")
        else:
            # 单数据集
            raw_dataset = load_from_disk(config.get("data_path"))
            if isinstance(raw_dataset, DatasetDict):
                if "validation" in raw_dataset:
                    train_data, val_data = raw_dataset["train"], raw_dataset["validation"]
                elif "test" in raw_dataset:
                    train_data, val_data = raw_dataset["train"], raw_dataset["test"]
                else:
                    split = raw_dataset["train"].train_test_split(test_size=0.05, seed=42)
                    train_data, val_data = split["train"], split["test"]
            elif isinstance(raw_dataset, Dataset):
                split = raw_dataset.train_test_split(test_size=0.05, seed=42)
                train_data, val_data = split["train"], split["test"]
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
        # ==========================================
    # 权重加载逻辑 (适配 MiniPile Init)
    # ==========================================
    start_step = 0
    
    # 1. 优先检查是否有显式指定的 Resume 路径
    resume_path = args.resume
    
    # 2. 如果没指定 resume，检查是否有 MiniPile 初始化权重 (清洗版)
    if not resume_path:
        # 假设你把清洗后的权重放在这里，名字固定
        minipile_init_path = f"checkpoints/{config['version']}_{config['scale']}/init.pth"
        if os.path.exists(minipile_init_path):
            print(f"✨ Found init checkpoint: {minipile_init_path}")
            resume_path = minipile_init_path
    
    checkpoint = None
    if resume_path and os.path.exists(resume_path):
        print(f"📦 Loading checkpoint from {resume_path}...")
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # strict=False 允许一些微小的 key 差异，但主要权重必须匹配
            model.load_state_dict(checkpoint['model'], strict=False)
            print("✅ Model weights loaded.")
            
            # 尝试加载优化器 (如果有)
            if 'optimizer' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print("✅ Optimizer state restored.")
                except Exception as e:
                    print(f"⚠️ Optimizer load failed (expected for init weights): {e}")
            else:
                print("ℹ️ No optimizer state found (Fresh start).")
            
            # 尝试恢复步数 (如果是 init 权重，step 应该是 0)
            if 'step' in checkpoint:
                start_step = checkpoint['step']
                # 如果是 step 40000 这种结束点，我们要强行重置为 0
                # 只有当它是中间存档时才继续
                if "init" in resume_path or start_step >= config['total_steps']:
                    print(f"🔄 Resetting step from {start_step} to 0 for new training phase.")
                    start_step = 0
                else:
                    start_step += 1
                    print(f"🔄 Resuming from step {start_step}")
        else:
            # 旧格式
            model.load_state_dict(checkpoint, strict=False)
            print("⚠️ Loaded weights only (Legacy format).")
            
    else:
        # 3. 既没 Resume 也没 Init，才去加载 RWKV 底模
        print("🌱 No checkpoint found. Loading RWKV backbone...")
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
    
    # ==========================================
    # SwanLab 初始化 (带图表续接功能)
    # ==========================================
    current_run_id = None
    run_id = None
    
    # 1. 如果是 Resume，尝试从 checkpoint 找 run_id
    if args.resume and isinstance(checkpoint, dict) and 'swanlab_run_id' in checkpoint:
        run_id = checkpoint['swanlab_run_id']
        print(f"🔄 Resuming SwanLab run: {run_id}")
    
    # 2. 初始化 SwanLab
    if HAS_SWANLAB:
        experiment = swanlab.init(
            project=config['project'],
            name=config['run_name'],
            config=config,
            id=run_id,
            resume="allow"
        )
        # 获取当前的 run_id (如果是新的，这里会生成新的)
        current_run_id = experiment.run.id
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"🚀 Training start from step {start_step}...")
    
    # ==========================================
    # Logging 逻辑 (回滚到瞬时值 + 修复Step显示)
    # ==========================================
    log_interval = config.get('log_interval', 10)
    
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
        if step % log_interval == 0:
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
            
            # 打印 (瞬时 Loss)
            log_str = f"Step {step} | Loss: {main_loss.item():.3f}"
            if val_loss:
                log_str += f" | ValLoss: {val_loss:.3f}"
            log_str += f" | Trans%: {trans_share:.1f} | TPS: {tps:.0f} | [{phase.upper()}]"
            print(log_str)
            
            # SwanLab Log (关键修正：传入 step 参数)
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
                
                # [关键] 显式指定 step，这样 step 1000 就会画在 X=1000 处
                swanlab.log(logs, step=step)
        
        # 保存完整 Checkpoint (顺便保存 run_id)
        if step > 0 and step % 2000 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 Cache cleared")
            path = os.path.join(config['save_dir'], f"{config['version']}_step{step}.pth")
        
            checkpoint_data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': config,
                'swanlab_run_id': current_run_id,
                'version': config['version']  # 额外记录版本
            }
            torch.save(checkpoint_data, path)
            print(f"💾 Saved: {path}")
    
    final_path = os.path.join(config['save_dir'], f"{config['version']}_final.pth")
    torch.save(
        {
            'model': model.state_dict(),
            'step': config['total_steps'],
            'config': config,
            'swanlab_run_id': current_run_id,
            'version': config['version'],
        },
        final_path
    )
    print("🎉 Done!")

if __name__ == "__main__":
    main()