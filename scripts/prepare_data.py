"""
CaMoE v18 Data Preprocessor (Ultimate Edition)
功能:
1. 加载多个数据源 (TinyStories, Ultrachat, Cosmo, MiniPile)
2. 清洗 & 格式化 (User/Assistant)
3. 采样 & 混合 (Interleave)
4. Tokenize & Packing (Rust RWKV Tokenizer)
5. 保存为单一数据集，供 train.py 直接读取
"""

import os
import argparse
import multiprocessing
from typing import Any, Dict
import re
from datasets import load_dataset, interleave_datasets
import pyrwkv_tokenizer

# ================= 配置 =================
# 定义你的配方 (Recipe)
# 格式: "name": (path_or_id, split, mode, probability)
# mode: "raw" (纯文本) 或 "chat" (对话)
DATA_RECIPE = {
    "tinystories": ("roneneldan/TinyStories", "train[:10%]", "raw", 0.4), # 取10%
    "cosmopedia":  ("HuggingFaceTB/cosmopedia-100k", "train", "raw", 0.3), # 全量
    "ultrachat":   ("HuggingFaceH4/ultrachat_200k", "train_sft", "chat", 0.2),
    "dailydialog": ("roskoN/dailydialog", "train", "chat", 0.1),
}

# 如果 Ultrachat 还是那个 list 格式，我们需要特殊处理
# 这里假设 Ultrachat 是标准的 HF 格式

def get_args() -> argparse.Namespace:
    r"""get_args() -> argparse.Namespace

    解析数据预处理命令行参数。

    Returns:
      argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./data/camoe_mix_v1", help="保存路径")
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=4, help="并行进程数，内存小设为2-4")
    parser.add_argument("--batch_size", type=int, default=100, help="Tokenize批次大小，内存小设为50")
    return parser.parse_args()

def process_text(item: Dict[str, Any], mode: str = "raw") -> str:
    r"""process_text(item, mode="raw") -> str

    将不同来源样本标准化为训练文本。

    Args:
      item (Dict[str, Any]): 单条样本。
      mode (str, optional): ``"raw"`` 或 ``"chat"``。Default: ``"raw"``。

    Returns:
      str: 清洗后的文本；无效样本返回空字符串。
    """
    text = ""
    
    # 1. 尝试获取内容
    # Ultrachat 200k: 'messages' (list[{"role","content"}])
    # Ultrachat(old): 'data' (list)
    # DailyDialog: 'dialog' (list)
    # TinyStories/Cosmo: 'text' (str)
    
    raw = None
    if 'messages' in item:
        raw = item['messages']
    elif 'text' in item:
        raw = item['text']
    elif 'data' in item:
        raw = item['data']
    elif 'dialog' in item:
        raw = item['dialog']
    
    if raw is None: return ""

    # 2. 格式化
    if isinstance(raw, list):
        # 对话列表 -> Chat 格式（兼容 list[str] / list[dict]）
        conversation = []
        for i, turn in enumerate(raw):
            if not turn:
                continue

            role = None
            content = None
            if isinstance(turn, dict):
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", "")).strip()
            else:
                content = str(turn).strip()

            if not content:
                continue

            content = content.replace('\r\n', '\n')
            content = re.sub(r'\n{2,}', '\n', content) # 去除多余换行

            if mode == "chat":
                if role in ("user", "assistant", "system"):
                    if role == "system":
                        # system 统一并入文本，但标注角色，便于模型感知
                        line = f"System: {content}"
                    elif role == "user":
                        line = f"User: {content}"
                    else:
                        line = f"Assistant: {content}"
                else:
                    # 无角色时回退到交替规则
                    guessed = "User" if i % 2 == 0 else "Assistant"
                    line = f"{guessed}: {content}"
                conversation.append(line)
            else:
                conversation.append(content)
        text = "\n\n".join(conversation)
        
    elif isinstance(raw, str):
        # 纯文本
        text = raw.strip().replace(" __eou__ ", "\n")
        if mode == "chat":
            # 如果是 Chat 模式但原始是文本，尝试转换(简易版)
            pass 
            
    return text

# 全局 Tokenizer (Worker 用)
tokenizer = None
def init_tokenizer():
    r"""init_tokenizer() -> None

    初始化全局 RWKV tokenizer，供多进程 worker 调用。
    """
    global tokenizer
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()

def tokenize_and_pack(batch: Dict[str, Any], ctx_len: int = 1024) -> Dict[str, Any]:
    r"""tokenize_and_pack(batch, ctx_len=1024) -> Dict[str, Any]

    对文本进行分词并打包成固定长度序列。

    Args:
      batch (Dict[str, Any]): batched 样本字典，需包含 ``text_processed``。
      ctx_len (int, optional): 序列长度。Default: ``1024``。

    Returns:
      Dict[str, Any]: 包含 ``input_ids`` 列的新批次。
    """
    global tokenizer
    if tokenizer is None:
        # 在 datasets 的多进程 map worker 中做懒加载初始化，避免 NoneType.encode 报错
        init_tokenizer()

    texts = batch['text_processed']
    if not texts: return {"input_ids": []}
    
    # 流式处理：边tokenize边pack，不缓存全部token
    chunks = []
    current_chunk = []
    
    for text in texts:
        ids = tokenizer.encode(text)
        ids.append(0)  # EOS
        
        for token in ids:
            current_chunk.append(token)
            if len(current_chunk) == ctx_len:
                chunks.append(current_chunk)
                current_chunk = []
        
        # 内存保护：限制缓存chunk数量
        if len(chunks) > 10000:
            break
    
    # 丢弃尾部不足ctx_len的token
    return {"input_ids": chunks}

def main() -> None:
    r"""main() -> None

    混合多源数据并导出为可直接训练的数据集格式。
    """
    args = get_args()
    print(f"🚀 Preparing Mixed Dataset -> {args.save_path}")
    
    datasets = []
    probs = []
    
    # 1. 加载并标准化所有源
    for name, (path, split, mode, prob) in DATA_RECIPE.items():
        print(f"  - Loading {name} ({split})...")
        try:
            ds = load_dataset(path, split=split)
            
            # Map: 统一转成 text_processed 列
            # 这里我们用单进程 map 快速处理文本格式化，或者多进程
            ds = ds.map(
                lambda x: {"text_processed": process_text(x, mode)},
                remove_columns=ds.column_names, # 移除原始列，只留 text_processed
                num_proc=args.num_proc,
                desc=f"Formatting {name}"
            )
            
            # 过滤空样本
            ds = ds.filter(lambda x: len(x['text_processed']) > 0)
            
            datasets.append(ds)
            probs.append(prob)
            print(f"    -> {len(ds)} samples ready.")
            
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
    
    if not datasets:
        print("❌ No datasets loaded!")
        return

    # 2. 混合 (Interleave)
    # 归一化概率
    total_p = sum(probs)
    probs = [p/total_p for p in probs]
    
    print(f"🥣 Mixing datasets with probs: {probs}...")
    mixed_ds = interleave_datasets(datasets, probabilities=probs, seed=42, stopping_strategy="first_exhausted")
    # 注意：first_exhausted 可能会丢弃大及其数据，all_exhausted 会过采样小数据
    # 对于预训练，通常用 probabilities 采样即可，不用太在意 epoch 边界
    # 如果想“存下来”，建议用 stopping_strategy="first_exhausted" 然后设个 limit?
    # 或者直接由 map 处理时就是流式的。
    
    # 这里的 mixed_ds 是 Lazy 的。
    
    # 3. Tokenize & Pack (最终处理)
    print("⚙️  Tokenizing & Packing (This will take a while)...")
    final_ds = mixed_ds.map(
        lambda x: tokenize_and_pack(x, args.ctx_len),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=["text_processed"],
        desc="Final Packing"
    )
    
    # 4. 保存
    print(f"💾 Saving to disk...")
    final_ds.save_to_disk(args.save_path)
    
    total_tokens = len(final_ds) * args.ctx_len
    print(f"✅ Done! Total Tokens: {total_tokens / 1e9:.4f} B")

if __name__ == "__main__":
    main()
