"""
CaMoE v20 训练脚本
支持: 七阶段调度 / 分组学习率 / 分阶段数据 profile / 经济系统增强 / Eval Loss
"""

import os
import gc
import time
import argparse
from typing import Dict, Iterator, List, Tuple, Any
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


def load_backbone(model: CaMoE_System, path: str) -> None:
    r"""load_backbone(model, path) -> None

    从 RWKV 底模迁移可对齐权重到 CaMoE 架构。

    Args:
      model (CaMoE_System): 当前 CaMoE 模型。
      path (str): RWKV 底模权重路径。
    """
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

def _build_phase_plan(config: Dict) -> List[Dict[str, Any]]:
    schedule = config.get("phase_schedule")
    if schedule:
        plan = []
        cursor = 0
        for phase in schedule:
            item = dict(phase)
            steps = max(0, int(item.get("steps", 0)))
            item["steps"] = steps
            item["start_step"] = cursor
            item["end_step"] = cursor + steps
            plan.append(item)
            cursor += steps
        return plan

    # 兼容旧配置：三阶段
    pre = int(config.get("prewarm_steps", 100))
    warm_end = int(config.get("warmup_steps", 500))
    warm = max(0, warm_end - pre)
    total = int(config.get("total_steps", warm_end + 1000))
    normal = max(0, total - warm_end)
    return [
        {
            "name": "prewarm",
            "steps": pre,
            "start_step": 0,
            "end_step": pre,
            "data_profile": "default",
            "train_groups": ["all"],
            "lr_mult": {g: 1.0 for g in config.get("param_groups", [])},
            "market_update": False,
        },
        {
            "name": "warmup",
            "steps": warm,
            "start_step": pre,
            "end_step": pre + warm,
            "data_profile": "default",
            "train_groups": ["all"],
            "lr_mult": {g: 1.0 for g in config.get("param_groups", [])},
            "market_update": False,
        },
        {
            "name": "normal",
            "steps": normal,
            "start_step": pre + warm,
            "end_step": pre + warm + normal,
            "data_profile": "default",
            "train_groups": ["all"],
            "lr_mult": {g: 1.0 for g in config.get("param_groups", [])},
            "market_update": True,
        },
    ]


def _phase_total_steps(phase_plan: List[Dict[str, Any]]) -> int:
    return int(sum(max(0, int(p.get("steps", 0))) for p in phase_plan))


def get_phase(step: int, phase_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    r"""get_phase(step, phase_plan) -> Dict"""
    for phase in phase_plan:
        if step < phase["end_step"]:
            return phase
    # 超范围时回退最后一个有步数的阶段
    for phase in reversed(phase_plan):
        if phase.get("steps", 0) > 0:
            return phase
    return phase_plan[-1] if phase_plan else {"name": "normal", "data_profile": "default", "market_update": True}


def _classify_param_group(name: str, num_rwkv: int) -> str:
    if ".critic." in name:
        return "critic"
    if name.startswith("bridge."):
        return "bridge"
    if name.startswith("emb.") or name.startswith("ln_out.") or name.startswith("head.") or name.startswith("deep_embed."):
        return "emb_head"

    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) > 3 and parts[2] == "experts":
            try:
                expert_idx = int(parts[3])
                return "rwkv_experts" if expert_idx < num_rwkv else "trans_experts"
            except ValueError:
                pass
        return "rwkv_backbone"

    return "rwkv_backbone"


def build_param_groups(model: CaMoE_System, config: Dict) -> Tuple[List[Dict[str, Any]], Dict[str, List[torch.nn.Parameter]]]:
    r"""build_param_groups(model, config) -> (optimizer_param_groups, group_map)"""
    groups = {g: [] for g in config.get("param_groups", [
        "rwkv_backbone", "rwkv_experts", "trans_experts", "bridge", "critic", "emb_head"
    ])}
    num_rwkv = int(config.get("num_rwkv_experts", 6))

    for name, p in model.named_parameters():
        g = _classify_param_group(name, num_rwkv)
        if g not in groups:
            groups[g] = []
        groups[g].append(p)

    base_lr = float(config.get("base_lr", 1e-4))
    optim_groups = []
    for gname, params in groups.items():
        if not params:
            continue
        optim_groups.append({
            "params": params,
            "lr": base_lr,
            "name": gname,
        })
    return optim_groups, groups


def apply_phase_policy(
    optimizer,
    phase: Dict[str, Any],
    config: Dict,
    group_map: Dict[str, List[torch.nn.Parameter]],
) -> None:
    r"""apply_phase_policy(optimizer, phase, config, group_map) -> None"""
    base_lr = float(config.get("base_lr", 1e-4))
    train_groups = set(phase.get("train_groups", ["all"]))
    train_all = "all" in train_groups
    lr_mult = phase.get("lr_mult", {})

    for gname, params in group_map.items():
        active = train_all or (gname in train_groups)
        for p in params:
            p.requires_grad = active

    for pg in optimizer.param_groups:
        gname = pg.get("name", "")
        mult = float(lr_mult.get(gname, 1.0 if (train_all or gname in train_groups) else 0.0))
        if not (train_all or gname in train_groups):
            mult = 0.0
        pg["lr"] = base_lr * mult


def apply_route_grad_policy(model: CaMoE_System, phase_name: str, config: Dict) -> None:
    r"""apply_route_grad_policy(model, phase_name, config) -> None

    criticwarm 期间开启 route grad（训练 critic），其它阶段恢复默认 route_no_grad。
    """
    default_route_no_grad = bool(config.get("route_no_grad", True))
    route_no_grad = default_route_no_grad
    if phase_name == "criticwarm":
        route_no_grad = False

    for block in model.blocks:
        block.route_no_grad = route_no_grad


def _load_profile_datasets(config: Dict, profile_name: str) -> Tuple[Dataset, Dataset]:
    data_profiles = config.get("data_profiles") or {}
    profile = data_profiles.get(profile_name, {})

    if profile_name == "default" and not profile:
        profile = {
            "mix": config.get("mix"),
            "data_path": config.get("data_path"),
        }

    mix = profile.get("mix")
    data_path = profile.get("data_path", config.get("data_path"))
    data_roots = config.get("data_roots") or {}

    if mix and data_roots:
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
            probs.append(float(prob))
            loaded_names.append(name)
            print(f"  - [{profile_name}] {name}: train={len(tr)}, val={len(va)} (prob={prob})")

        if not train_datasets:
            raise ValueError(f"No valid datasets in profile '{profile_name}'.")

        total_p = sum(probs)
        probs = [p / total_p for p in probs]
        train_data = interleave_datasets(
            train_datasets,
            probabilities=probs,
            seed=42,
            stopping_strategy="all_exhausted",
        )
        val_data = interleave_datasets(
            val_datasets,
            probabilities=probs,
            seed=42,
            stopping_strategy="first_exhausted",
        )
        print(f"📊 Profile={profile_name} Mix: {dict(zip(loaded_names, probs))} -> Train={len(train_data)}, Val={len(val_data)}")
        return train_data, val_data

    # 单数据集
    if not data_path:
        raise ValueError(f"Profile '{profile_name}' has no mix and no data_path.")
    raw_dataset = load_from_disk(data_path)
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
    print(f"📊 Profile={profile_name}: Train={len(train_data)}, Val={len(val_data)}")
    return train_data, val_data


def build_loader_for_profile(
    profile_name: str,
    config: Dict,
    collate_fn,
) -> Tuple[DataLoader, DataLoader]:
    r"""build_loader_for_profile(profile_name, config, collate_fn) -> (train_loader, val_loader)"""
    train_data, val_data = _load_profile_datasets(config, profile_name)
    train_loader = DataLoader(
        train_data,
        batch_size=config["micro_batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config["micro_batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader

def log_gpu() -> str:
    r"""log_gpu() -> str

    返回当前 GPU 显存占用摘要字符串。

    Returns:
      str: 显存信息；无 CUDA 时返回空字符串。
    """
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"GPU: {alloc:.1f}/{total:.1f}GB"
    return ""

def infinite_loader(loader: DataLoader) -> Iterator:
    r"""infinite_loader(loader) -> Iterator

    将有限 DataLoader 包装为无限迭代器。

    Args:
      loader (DataLoader): 原始数据加载器。

    Returns:
      Iterator: 循环产出 batch 的迭代器。
    """
    while True:
        for batch in loader:
            yield batch

def main() -> None:
    r"""main() -> None

    训练主入口，包含数据加载、断点续训、阶段训练、评估与保存。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="0.4b")
    parser.add_argument(
        "--diag",
        default="baseline",
        choices=["baseline", "no_amp", "no_fast_math", "fp32_kernel"],
        help="Diagnostic switches for isolating NaN source",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config = get_config(args.scale)

    # Diagnostic overrides (switch-only; no model structure changes)
    if args.diag == "no_amp":
        config["train_use_amp"] = False
    elif args.diag == "no_fast_math":
        config["cuda_use_fast_math"] = False
    elif args.diag == "fp32_kernel":
        config["cuda_use_fast_math"] = False
        config["cuda_force_fp32_kernel"] = True

    # Pass diagnostic kernel switches through env for backbone.init_rwkv7_cuda()
    os.environ["CAMOE_DISABLE_FAST_MATH"] = "1" if not config.get("cuda_use_fast_math", True) else "0"
    os.environ["CAMOE_FORCE_FP32_KERNEL"] = "1" if config.get("cuda_force_fp32_kernel", False) else "0"
    os.environ["CAMOE_NAN_DEBUG"] = "1" if config.get("nan_debug", False) else "0"
    os.environ["CAMOE_SANITIZE_TIMEMIX_OUT"] = "1" if config.get("sanitize_timemix_output", False) else "0"
    os.environ["CAMOE_FORCE_TIMEMIX_FALLBACK"] = "1" if config.get("force_timemix_fallback", False) else "0"
    init_rwkv7_cuda()
    
    # 强制设置 Eval 频率
    eval_interval = config.get('eval_interval', 1000)  # 每500步评测一次
    eval_iters = config.get('eval_iters', 50)         # 每次评测跑50个batch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    train_use_amp = bool(config.get("train_use_amp", True)) and device == "cuda"
    amp_dtype_name = str(config.get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    print(
        f"🧪 Diag={args.diag} | AMP={train_use_amp}({amp_dtype_name}) | "
        f"fast_math={config.get('cuda_use_fast_math', True)} | "
        f"fp32_kernel={config.get('cuda_force_fp32_kernel', False)}"
    )

    phase_plan = _build_phase_plan(config)
    config["phase_schedule"] = phase_plan
    total_steps_from_schedule = _phase_total_steps(phase_plan)
    if total_steps_from_schedule > 0:
        config["total_steps"] = total_steps_from_schedule
    phase_to_id = {p.get("name", f"phase_{i}"): i for i, p in enumerate(phase_plan)}

    # 2. DataLoader Collate
    def simple_collate(batch) -> torch.Tensor:
        r"""simple_collate(batch) -> Tensor

        将变长样本拼接为固定长度 batch，并按 CUDA kernel 约束进行长度对齐。

        Args:
          batch: 原始样本列表，每项包含 ``input_ids``。

        Returns:
          Tensor: 形状 ``[B, T]`` 的 long 张量。
        """
        input_ids = [item["input_ids"] for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, config['ctx_len'] + 1)
        
        # [CUDA Kernel 要求] 对齐到 16 的倍数 + 1
        CHUNK_LEN = 16
        input_len = ((max_len - 1 + CHUNK_LEN - 1) // CHUNK_LEN) * CHUNK_LEN
        target_len = max(input_len + 1, CHUNK_LEN + 1)
        
        padded_batch = torch.full((len(batch), target_len), -100, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            l = min(len(ids), target_len)
            padded_batch[i, :l] = ids[:l]
        return padded_batch

    # 3. Model & Optimizer
    print("🏗️ Building model...")
    model = CaMoE_System(config).to(device)

    optimizer_groups, group_map = build_param_groups(model, config)
    optimizer = bnb.optim.AdamW8bit(optimizer_groups, lr=config.get("base_lr", 1e-4))

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
    # 按当前 step 对齐阶段与数据 profile
    # ==========================================
    current_phase = get_phase(start_step, phase_plan)
    current_profile = current_phase.get("data_profile", "default")
    print(f"🚀 Loading datasets for phase={current_phase.get('name')} profile={current_profile} ...")
    try:
        train_loader, val_loader = build_loader_for_profile(current_profile, config, simple_collate)
    except Exception as e:
        print(f"❌ Error loading dataset profile '{current_profile}': {e}")
        return
    train_iter = infinite_loader(train_loader)
    
    # ==========================================
    # [新增] 评估函数
    # ==========================================
    @torch.no_grad()
    def estimate_loss(model: CaMoE_System, loader: DataLoader, eval_steps: int) -> float:
        r"""estimate_loss(model, loader, eval_steps) -> float

        在验证集上估算平均交叉熵损失。

        Args:
          model (CaMoE_System): 待评估模型。
          loader (DataLoader): 验证集加载器。
          eval_steps (int): 评估批次数。

        Returns:
          float: 平均验证损失；若无有效值则返回 ``inf``。
        """
        model.eval()
        losses = []
        
        # 使用 itertools.cycle 无限循环验证集，避免 StopIteration
        from itertools import cycle
        
        for i, batch in enumerate(cycle(loader)):
            if i >= eval_steps:
                break
            
            batch = batch.to(device)
            if batch.shape[1] <= 1: 
                continue
            
            x, y = batch[:, :-1], batch[:, 1:]
            
            # Eval 时使用 Normal 模式，测试全系统
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=train_use_amp):
                logits, info = model(x, step=100000, phase="normal")
                # 只算 Main Loss
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    y.reshape(-1),
                    ignore_index=-100,
                )
            
            losses.append(loss.item())
            
            # 安全检查：如果损失为 NaN 或 Inf，立即报告
            if not torch.isfinite(torch.tensor(loss.item())):
                print(f"⚠️ Invalid loss detected at eval step {i}: {loss.item()}")
                continue
        
        model.train()
        if len(losses) == 0:
            print("⚠️ Warning: No valid losses collected during evaluation!")
            return float('inf')
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
        current_run_id = experiment.public.run_id
    
    os.makedirs(config['save_dir'], exist_ok=True)
    
    print(f"🚀 Training start from step {start_step}...")
    
    # ==========================================
    # Logging 逻辑 (回滚到瞬时值 + 修复Step显示)
    # ==========================================
    log_interval = config.get('log_interval', 10)
    last_phase_name = None
    active_profile = current_profile
    
    # 5. Training Loop
    for step in range(start_step, config['total_steps']):
        t0 = time.time()
        
        phase = get_phase(step, phase_plan)
        phase_name = phase.get("name", "normal")
        if phase_name != last_phase_name:
            apply_phase_policy(optimizer, phase, config, group_map)
            apply_route_grad_policy(model, phase_name, config)
            last_phase_name = phase_name
            print(f"🔁 Phase switched -> {phase_name} [{phase.get('start_step', step)}:{phase.get('end_step', step)}]")

            phase_profile = phase.get("data_profile", "default")
            if phase_profile != active_profile:
                print(f"🧭 Rebuilding dataloaders for profile={phase_profile}")
                train_loader, val_loader = build_loader_for_profile(phase_profile, config, simple_collate)
                train_iter = infinite_loader(train_loader)
                active_profile = phase_profile
        
        try:
            x_batch = next(train_iter)
        except StopIteration:
            train_iter = infinite_loader(train_loader)
            x_batch = next(train_iter)
            
        x_batch = x_batch.to(device)
        if x_batch.shape[1] <= 1: continue
            
        x, y = x_batch[:, :-1], x_batch[:, 1:]
        
        try:
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=train_use_amp):
                logits, info = model(x, step=step, phase=phase_name)
                total_loss, token_losses, main_loss, critic_loss, _bridge_loss = model.compute_losses(logits, y, info)
                loss_to_backward = total_loss / config['grad_accum']
        except RuntimeError as e:
            print(f"💥 Forward/LOSS failed at step={step}, phase={phase_name}: {e}")
            raise

        if not torch.isfinite(loss_to_backward):
            print(f"⚠️ Non-finite loss at step {step}: {float(loss_to_backward)} (skip batch)")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_to_backward.backward()
        
        if (step + 1) % config['grad_accum'] == 0:
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if phase.get("market_update", True) and step > 100:
                model.update_market(info, token_losses, step, phase=phase_name, critic_loss=critic_loss)
        
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
            
            # 打印 (瞬时 Loss)
            log_str = f"Step {step} | Loss: {main_loss.item():.3f}"
            if val_loss:
                log_str += f" | ValLoss: {val_loss:.3f}"
            log_str += f" | TPS: {tps:.0f} | [{phase_name.upper()}]"
            print(log_str)
            
            # SwanLab Log (关键修正：传入 step 参数)
            if HAS_SWANLAB:
                logs = {
                    "Loss/Train_Main": main_loss.item(),
                    "Loss/Train_Critic": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
                    "Speed/TPS": tps,
                    "Phase/ID": float(phase_to_id.get(phase_name, -1)),
                    f"Phase/{phase_name}": 1.0,
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
