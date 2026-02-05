# analyze_checkpoints.py
import os
import glob
import torch
from camoe.config import CONFIG_MINIPILE

def analyze_checkpoint(ckpt_path):
    """分析单个 checkpoint 的市场状态"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return None
    
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
        step = ckpt.get('step', '?')
    else:
        state_dict = ckpt
        # 从文件名猜步数
        import re
        match = re.search(r'step(\d+)', ckpt_path)
        step = int(match.group(1)) if match else '?'
    
    # 找 capitals
    capitals_key = None
    for k in state_dict.keys():
        if 'capitals' in k and 'capital_manager' in k:
            capitals_key = k
            break
    
    if capitals_key is None:
        return {'step': step, 'error': 'capitals not found'}
    
    capitals = state_dict[capitals_key]  # [n_layer, n_experts]
    n_layers, n_experts = capitals.shape
    
    result = {
        'step': step,
        'path': os.path.basename(ckpt_path),
        'n_layers': n_layers,
        'n_experts': n_experts,
        'layers': {}
    }
    
    for layer_idx in range(n_layers):
        caps = capitals[layer_idx]
        
        # 计算各种指标
        shares = caps / (caps.sum() + 1e-6) * 100  # 百分比
        
        # Gini
        sorted_caps, _ = torch.sort(caps)
        n = n_experts
        idx = torch.arange(1, n + 1, dtype=caps.dtype)
        gini = ((2 * idx - n - 1) * sorted_caps).sum() / (n * caps.sum() + 1e-6)
        
        # 最大/最小专家
        max_idx = caps.argmax().item()
        min_idx = caps.argmin().item()
        
        result['layers'][layer_idx] = {
            'gini': gini.item(),
            'shares': shares.tolist(),
            'max_expert': max_idx,
            'max_share': shares[max_idx].item(),
            'min_expert': min_idx,
            'min_share': shares[min_idx].item(),
            'capitals': caps.tolist(),
        }
    
    return result


def print_analysis(results):
    """打印分析结果"""
    print("\n" + "=" * 80)
    print("📊 CHECKPOINT ANALYSIS REPORT")
    print("=" * 80)
    
    for r in results:
        if r is None:
            continue
        if 'error' in r:
            print(f"\n❌ {r['path']}: {r['error']}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📦 {r['path']} (Step {r['step']})")
        print(f"   Layers: {r['n_layers']}, Experts: {r['n_experts']}")
        print("-" * 80)
        
        # 表头
        print(f"{'Layer':>6} | {'Gini':>6} | {'Expert Shares (%)':^40} | {'Winner':>8}")
        print("-" * 80)
        
        for layer_idx, data in r['layers'].items():
            shares_str = " ".join([f"E{i}:{s:5.1f}" for i, s in enumerate(data['shares'])])
            winner = f"E{data['max_expert']}({data['max_share']:.1f}%)"
            
            # Gini 颜色标记
            gini = data['gini']
            if gini < 0.1:
                gini_mark = "⚪"  # 太平均
            elif gini < 0.3:
                gini_mark = "🟢"  # 健康
            elif gini < 0.5:
                gini_mark = "🟡"  # 有分化
            else:
                gini_mark = "🔴"  # 高度不平等
            
            print(f"  L{layer_idx:>3} | {gini:>5.3f}{gini_mark} | {shares_str} | {winner}")
        
        # 汇总
        avg_gini = sum(d['gini'] for d in r['layers'].values()) / len(r['layers'])
        print("-" * 80)
        print(f"  平均 Gini: {avg_gini:.3f}")
        
        # 检查 Transformer 专家（假设是最后几个）
        num_rwkv = CONFIG_MINIPILE.get('num_rwkv_experts', 2)
        num_trans = CONFIG_MINIPILE.get('num_trans_experts', 1)
        
        trans_shares = []
        for layer_idx, data in r['layers'].items():
            shares = data['shares']
            trans_total = sum(shares[num_rwkv:])  # Transformer 专家总份额
            trans_shares.append(trans_total)
        
        avg_trans = sum(trans_shares) / len(trans_shares)
        print(f"  Transformer 专家平均份额: {avg_trans:.1f}%")


def recommend_checkpoints(results):
    """推荐重点测试的 checkpoint"""
    print("\n" + "=" * 80)
    print("🎯 推荐测试的 CHECKPOINT")
    print("=" * 80)
    
    valid_results = [r for r in results if r and 'error' not in r]
    
    if not valid_results:
        print("没有有效的 checkpoint!")
        return
    
    # 按步数排序
    valid_results.sort(key=lambda x: x['step'] if isinstance(x['step'], int) else 0)
    
    recommendations = []
    
    # 1. 最新的
    latest = valid_results[-1]
    recommendations.append(('📍 最新', latest))
    
    # 2. Gini 最健康的（0.2-0.4 之间）
    def gini_health(r):
        avg_gini = sum(d['gini'] for d in r['layers'].values()) / len(r['layers'])
        return abs(avg_gini - 0.3)  # 越接近 0.3 越好
    
    healthiest = min(valid_results, key=gini_health)
    if healthiest != latest:
        recommendations.append(('🏥 Gini 最健康', healthiest))
    
    # 3. Transformer 份额最高的
    num_rwkv = CONFIG_MINIPILE.get('num_rwkv_experts', 2)
    def trans_share(r):
        total = 0
        for data in r['layers'].values():
            total += sum(data['shares'][num_rwkv:])
        return total / len(r['layers'])
    
    highest_trans = max(valid_results, key=trans_share)
    if highest_trans not in [r[1] for r in recommendations]:
        recommendations.append(('🤖 Transformer 最活跃', highest_trans))
    
    # 4. 中间点
    if len(valid_results) >= 3:
        mid = valid_results[len(valid_results) // 2]
        if mid not in [r[1] for r in recommendations]:
            recommendations.append(('📊 中间点', mid))
    
    # 打印推荐
    for label, r in recommendations:
        avg_gini = sum(d['gini'] for d in r['layers'].values()) / len(r['layers'])
        avg_trans = trans_share(r)
        print(f"\n{label}:")
        print(f"  📦 {r['path']} (Step {r['step']})")
        print(f"  📈 平均 Gini: {avg_gini:.3f}")
        print(f"  🤖 Transformer 份额: {avg_trans:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="checkpoints/minipile", help="Checkpoint 目录")
    parser.add_argument("--pattern", "-p", default="*.pth", help="文件匹配模式")
    args = parser.parse_args()
    
    # 找所有 checkpoint
    pattern = os.path.join(args.dir, args.pattern)
    ckpt_files = sorted(glob.glob(pattern))
    
    if not ckpt_files:
        print(f"❌ 没找到 checkpoint: {pattern}")
        return
    
    print(f"🔍 找到 {len(ckpt_files)} 个 checkpoint")
    
    # 分析每个
    results = []
    for path in ckpt_files:
        print(f"  分析: {os.path.basename(path)}...")
        result = analyze_checkpoint(path)
        results.append(result)
    
    # 打印分析
    print_analysis(results)
    
    # 推荐
    recommend_checkpoints(results)
    
    # 输出简洁版本供复制
    print("\n" + "=" * 80)
    print("📋 快速复制 (用于评测)")
    print("=" * 80)
    
    valid = [r for r in results if r and 'error' not in r]
    valid.sort(key=lambda x: x['step'] if isinstance(x['step'], int) else 0)
    
    for r in valid:
        avg_gini = sum(d['gini'] for d in r['layers'].values()) / len(r['layers'])
        print(f"Step {r['step']:>6} | Gini {avg_gini:.3f} | {r['path']}")


if __name__ == "__main__":
    main()