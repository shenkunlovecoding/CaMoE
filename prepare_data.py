import os
import sys
import argparse
import multiprocessing
from datasets import load_dataset, Dataset, DatasetDict

# =================配置区域=================
# 强制使用镜像站，防止网络报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# =========================================

# 导入 Tokenizer
sys.setrecursionlimit(10000)
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "tokenizer"))
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
except ImportError:
    # 尝试直接导入（适配不同目录结构）
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

def get_args():
    parser = argparse.ArgumentParser(description="通用数据集处理脚本 (MiniPile/SlimPajama)")
    parser.add_argument("--token",type=str,help="HF Token")
    # 核心参数
    parser.add_argument("--dataset", type=str, default="JeanKaddour/minipile", 
                        help="HuggingFace 数据集名称 (例如: JeanKaddour/minipile 或 Cerebras/SlimPajama-627B)")
    parser.add_argument("--split", type=str, default="train", 
                        help="要处理的分支 (train, validation, test)")
    parser.add_argument("--name", type=str, default=None,
                        help="数据集的子配置名称 (例如 SlimPajama 可能需要)")
    
    # 路径参数
    parser.add_argument("--save_path", type=str, default="./data/minipile_processed", 
                        help="处理后的数据保存路径")
    parser.add_argument("--vocab", type=str, default="tokenizer/rwkv_vocab_v20230424.txt", 
                        help="词表路径")
    
    # 处理参数
    parser.add_argument("--ctx_len", type=int, default=1024, help="上下文长度 (Chunk Size)")
    parser.add_argument("--num_proc", type=int, default=16, help="进程数 (9950X 建议 16-24)")
    parser.add_argument("--batch_size", type=int, default=1000, help="批处理大小")

    return parser.parse_args()

# 全局变量 (用于多进程)
global_tokenizer = None
CTX_LEN = 1024

def init_worker(vocab_path, ctx_len):
    """多进程初始化函数"""
    global global_tokenizer, CTX_LEN
    global_tokenizer = TRIE_TOKENIZER(vocab_path)
    CTX_LEN = ctx_len

def process_and_pack(examples):
    # 【Windows 防呆补丁】
    global global_tokenizer, CTX_LEN
    if global_tokenizer is None:
        # 如果子进程里是空的，现充一个
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        # 路径可能需要写死或者传参，这里假设在默认位置
        global_tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
        CTX_LEN = 1024 # 默认值
    """
    核心处理逻辑：
    1. 找到文本列
    2. Tokenize + EOS
    3. Packing (拼接后切分)
    """
    # 自动寻找文本列名 (有的数据集叫 text, 有的叫 content)
    text_column = "text"
    if "text" not in examples:
        if "content" in examples:
            text_column = "content"
        else:
            # 盲猜第一个全是字符串的列
            text_column = list(examples.keys())[0]

    all_token_ids = []
    
    for text in examples[text_column]:
        if not text or not isinstance(text, str) or not text.strip():
            continue
        
        # 编码并添加 EOS (0)
        # 注意：RWKV Tokenizer 在多进程下可能需要异常捕获
        try:
            ids = global_tokenizer.encode(text)
            if ids:
                all_token_ids.extend(ids + [0])
        except:
            continue
            
    # Packing: 切分成固定长度
    output = []
    for i in range(0, len(all_token_ids), CTX_LEN):
        chunk = all_token_ids[i : i + CTX_LEN]
        # 只保留完整的块，丢弃最后一点点尾巴 (数据量够大时可忽略)
        if len(chunk) == CTX_LEN:
            output.append(chunk)
            
    return {"input_ids": output}

def main():
    args = get_args()
    
    print(f"🚀 准备处理数据集: {args.dataset} [{args.split}]")
    print(f"📂 保存路径: {args.save_path}")
    print(f"🧵 进程数: {args.num_proc} | Context: {args.ctx_len}")

    # 1. 加载数据集
    print("☁️  正在从 HuggingFace 下载/加载数据...")
    try:
        if args.name:
            ds = load_dataset(args.dataset, args.name, split=args.split,token=args.token)
        else:
            ds = load_dataset(args.dataset, split=args.split)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print(f"📊 原始数据量: {len(ds)} 行")

    # 2. 并行处理
    print("⚙️  开始 Tokenizing & Packing...")
    
    # 你的 9950X 有 32 线程，这里开 16-24 个比较合适，留点给系统 IO
    n_proc = min(os.cpu_count(), args.num_proc)
    
    tokenized_dataset = ds.map(
        process_and_pack,
        batched=True,
        batch_size=args.batch_size,
        num_proc=n_proc,
        remove_columns=ds.column_names, # 移除原始文本列，只留 input_ids
        fn_kwargs={}, # 不需要传参，通过 init_worker 初始化全局变量
        load_from_cache_file=True, # 启用缓存，防止崩了重跑
        desc="Processing"
    )

    # 3. 保存
    print(f"💾 正在保存到磁盘: {args.save_path}")
    tokenized_dataset.save_to_disk(args.save_path)
    
    # 4. 统计
    final_tokens = len(tokenized_dataset) * args.ctx_len
    print(f"✅ 全部完成！")
    print(f"📊 最终样本数: {len(tokenized_dataset)}")
    print(f"📊 有效 Token 总量: {final_tokens / 1e9:.4f} B Tokens")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # 获取参数并初始化全局变量 (这步很关键，要在 map 之前做)
    args = get_args()
    
    # Hack: 这种写法在 Windows spawn 模式下可能需要特殊处理
    # 但 datasets 的 map 函数有自己的初始化机制，
    # 我们这里通过闭包或者简单的全局设置来做
    # 最稳妥的方式是把 init 逻辑放进 map 内部，或者利用全局作用域
    
    # 这里我们手动设置一下全局变量，供 Windows 子进程 fork/spawn 时使用
    # 注意：Windows 下 spawn 会重新 import 这个文件，所以需要在 if __name__ 外面也有一部分逻辑
    # 或者简单点，直接在 process_and_pack 里容错。
    
    # 修正：datasets 库在 Windows 下传递 Tokenizer 对象会很难受（pickle 问题）。
    # 最好的办法是让每个子进程自己重新加载 Tokenizer。
    # 我们使用 .map 的 new_fingerprint 参数或者 init 技巧，
    # 但最简单的就是利用 `process_and_pack` 里的 global_tokenizer。
    # 为了让它生效，我们需要一个 wrapper。
    
    # 重新定义带 init 的 map 调用：
    # 实际上 datasets 库目前处理多进程传参比较智能，只要 global_tokenizer 能被 pickle 即可。
    # RWKV 的 TRIE_TOKENIZER 应该没问题。
    
    # 手动初始化主进程的 tokenizer
    init_worker(args.vocab, args.ctx_len)
    
    main()