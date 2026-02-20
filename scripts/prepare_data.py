"""
CaMoE v20 Data Preprocessor (non-streaming)
流程:
1) 加载数据集
2) 文本标准化
3) 过滤空样本
4) 按配方混合
5) tokenize + pack
6) save_to_disk
"""

import argparse
import re
import time
from typing import Any, Dict

from datasets import interleave_datasets, load_dataset
import pyrwkv_tokenizer

# ================= 配置 =================
# (path_or_id, subset_or_none, split, mode, probability)
DATA_RECIPE = {
    # v20: FineWeb-Edu(sample-10BT) 70% + Cosmopedia-100k 30%
    "fineweb_edu": ("roneneldan/TinyStories", None, "train", "raw", 1),
}


def stage_log(stage: str, msg: str = "") -> float:
    now = time.perf_counter()
    line = f"\n========== [{stage}] =========="
    if msg:
        line += f"\n{msg}"
    print(line)
    return now


def stage_done(stage: str, t0: float) -> None:
    dt = time.perf_counter() - t0
    print(f"[{stage}] done in {dt:.2f}s")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./data/camoe_mix_v20_fineweb70_cosmo30", help="输出目录")
    parser.add_argument("--ctx_len", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=4, help="map/filter 并行进程")
    parser.add_argument("--batch_size", type=int, default=100, help="tokenize map 的 batched batch size")
    parser.add_argument("--seed", type=int, default=42, help="interleave 随机种子")
    return parser.parse_args()


def process_text(item: Dict[str, Any], mode: str = "raw", source: str = "") -> str:
    text = ""
    source_l = source.lower()

    # Cosmopedia: prompt + text
    if source_l.startswith("cosmopedia"):
        prompt = str(item.get("prompt", "")).strip()
        text = str(item.get("text", "")).strip()
        merged = text if not prompt else (f"{prompt}\n\n{text}" if text else prompt)
        return merged

    # FineWeb-Edu: 轻质控
    if source_l.startswith("fineweb"):
        lang_score = item.get("language_score", None)
        token_count = item.get("token_count", None)
        text = str(item.get("text", "")).strip()
        if lang_score is not None:
            try:
                if float(lang_score) < 0.6:
                    return ""
            except Exception:
                pass
        if token_count is not None:
            try:
                if int(token_count) < 32:
                    return ""
            except Exception:
                pass
        return text

    raw = None
    if "messages" in item:
        raw = item["messages"]
    elif "text" in item:
        raw = item["text"]
    elif "data" in item:
        raw = item["data"]
    elif "dialog" in item:
        raw = item["dialog"]
    if raw is None:
        return ""

    if isinstance(raw, list):
        conversation = []
        for i, turn in enumerate(raw):
            if not turn:
                continue
            if isinstance(turn, dict):
                role = str(turn.get("role", "")).strip().lower()
                content = str(turn.get("content", "")).strip()
            else:
                role = ""
                content = str(turn).strip()
            if not content:
                continue
            content = content.replace("\r\n", "\n")
            content = re.sub(r"\n{2,}", "\n", content)
            if mode == "chat":
                if role == "system":
                    line = f"System: {content}"
                elif role == "user":
                    line = f"User: {content}"
                elif role == "assistant":
                    line = f"Assistant: {content}"
                else:
                    guessed = "User" if i % 2 == 0 else "Assistant"
                    line = f"{guessed}: {content}"
                conversation.append(line)
            else:
                conversation.append(content)
        text = "\n\n".join(conversation)
    elif isinstance(raw, str):
        text = raw.strip().replace(" __eou__ ", "\n")

    return text


tokenizer = None


def init_tokenizer() -> None:
    global tokenizer
    tokenizer = pyrwkv_tokenizer.RWKVTokenizer()


def tokenize_and_pack(batch: Dict[str, Any], ctx_len: int = 1024) -> Dict[str, Any]:
    global tokenizer
    if tokenizer is None:
        init_tokenizer()

    texts = batch["text_processed"]
    if not texts:
        return {"input_ids": []}

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

        # 防止单 worker 临时峰值过高
        if len(chunks) > 10000:
            break

    return {"input_ids": chunks}


def main() -> None:
    args = get_args()
    print("🚀 PrepareData started")
    print(f"📁 save_path={args.save_path}")
    print(f"🧩 recipe={ {k: v[-1] for k, v in DATA_RECIPE.items()} }")
    print(f"⚙️ ctx_len={args.ctx_len}, num_proc={args.num_proc}, batch_size={args.batch_size}, seed={args.seed}")

    datasets = []
    probs = []

    t = stage_log("STAGE-1 LOAD+FORMAT")
    for name, recipe in DATA_RECIPE.items():
        if len(recipe) == 4:
            path, split, mode, prob = recipe
            subset = None
        elif len(recipe) == 5:
            path, subset, split, mode, prob = recipe
        else:
            print(f"⚠️ Invalid recipe format for {name}: {recipe}")
            continue

        print(f"  -> loading {name}: path={path} subset={subset} split={split} mode={mode} prob={prob}")
        try:
            if subset is None:
                ds = load_dataset(path, split=split)
            else:
                ds = load_dataset(path, subset, split=split)
            print(f"     raw samples: {len(ds)}")

            ds = ds.map(
                lambda x: {"text_processed": process_text(x, mode, name)},
                remove_columns=ds.column_names,
                num_proc=args.num_proc,
                desc=f"[{name}] formatting",
            )
            ds = ds.filter(
                lambda x: len(x["text_processed"]) > 0,
                num_proc=args.num_proc,
                desc=f"[{name}] filtering-empty",
            )
            print(f"     cleaned samples: {len(ds)}")
            datasets.append(ds)
            probs.append(float(prob))
        except Exception as e:
            print(f"⚠️ Failed to load {name}: {e}")
    stage_done("STAGE-1 LOAD+FORMAT", t)

    if not datasets:
        print("❌ No datasets loaded.")
        return

    t = stage_log("STAGE-2 MIX")
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    print(f"🥣 normalized probs={probs}")
    mixed_ds = interleave_datasets(
        datasets,
        probabilities=probs,
        seed=args.seed,
        stopping_strategy="first_exhausted",
    )
    print(f"📊 mixed samples={len(mixed_ds)}")
    stage_done("STAGE-2 MIX", t)

    t = stage_log("STAGE-3 TOKENIZE+PACK")
    final_ds = mixed_ds.map(
        lambda x: tokenize_and_pack(x, args.ctx_len),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=["text_processed"],
        desc="tokenize+pack",
    )
    print(f"📦 packed sequences={len(final_ds)}")
    stage_done("STAGE-3 TOKENIZE+PACK", t)

    t = stage_log("STAGE-4 SAVE")
    final_ds.save_to_disk(args.save_path)
    stage_done("STAGE-4 SAVE", t)

    total_tokens = len(final_ds) * args.ctx_len
    print(f"\n✅ Done. total_tokens={total_tokens / 1e9:.4f}B")


if __name__ == "__main__":
    main()
