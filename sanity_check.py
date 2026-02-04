# test_lambada_logic.py
import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from camoe import CaMoE_System
from config import CONFIG_BABYLM
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from backbone import init_rwkv7_cuda

init_rwkv7_cuda()

model = CaMoE_System(CONFIG_BABYLM).cuda().eval()
ckpt = torch.load("checkpoints/babylm/v12_step24000.pth", map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)
tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

# 模拟 LAMBADA 样本
test_cases = [
    ("The capital of France is", " Paris"),
    ("The color of the sky is", " blue"),
    ("One plus one equals", " two"),
    ("She picked up the phone and said", " hello"),
]

print("=" * 60)
print("📊 LAMBADA 风格测试")
print("=" * 60)

correct = 0
for context, target in test_cases:
    # Tokenize
    ctx_ids = tokenizer.encode(context)
    target_ids = tokenizer.encode(target)
    full_ids = ctx_ids + target_ids
    
    # Pad
    while len(full_ids) % 16 != 0:
        full_ids.append(0)
    
    with torch.no_grad():
        x = torch.tensor([full_ids], device='cuda')
        logits, _ = model(x, step=0, phase="normal")
    
    # 预测最后一个 context token 之后的词
    pred_pos = len(ctx_ids) - 1
    pred_logits = logits[0, pred_pos]
    
    # Top 5 预测
    probs = F.softmax(pred_logits.float(), dim=-1)
    top5 = torch.topk(probs, 5)
    
    # 检查是否命中
    predicted_token = pred_logits.argmax().item()
    actual_token = target_ids[0] if target_ids else -1
    is_correct = (predicted_token == actual_token)
    correct += int(is_correct)
    
    mark = "✅" if is_correct else "❌"
    print(f"\n{mark} Context: '{context}'")
    print(f"   Target: '{target}' (token: {actual_token})")
    print(f"   Predicted token: {predicted_token} = '{tokenizer.decode([predicted_token])}'")
    print(f"   Top 5:")
    for p, idx in zip(top5.values, top5.indices):
        tok = tokenizer.decode([idx.item()])
        hit = "←" if idx.item() == actual_token else ""
        print(f"      {p.item()*100:5.1f}% - '{tok}' {hit}")

print(f"\n{'='*60}")
print(f"📊 准确率: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.0f}%")
print(f"{'='*60}")