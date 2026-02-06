import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def train_custom_tokenizer():
    # 1. 准备路径
    save_path = "tokenizer/minipile_32k"
    os.makedirs(save_path, exist_ok=True)
    
    # 2. 初始化 Tokenizer (BPE 模式，类似 GPT-2/RoBERTa)
    print("⚙️  Initializing Tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    
    # 预处理：按字节切分 (ByteLevel)，这对代码和多语言支持很好
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 解码器
    tokenizer.decoder = decoders.ByteLevel()
    
    # 后处理 (RoBERTa 风格)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # 3. 设置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=32000,           # ✅ 你的目标：32k 词表
        min_frequency=2,            # 过滤掉只出现一次的词
        special_tokens=["<|endoftext|>", "<|padding|>"], # 特殊 Token
        show_progress=True
    )
    
    # 4. 加载数据迭代器 (流式加载，不占内存)
    print("☁️  Loading MiniPile dataset (streaming)...")
    dataset = load_dataset("JeanKaddour/minipile", split="train", streaming=True)
    
    def batch_iterator(batch_size=10000):
        for i, item in enumerate(dataset):
            yield item["text"]
            if i > 200_000: # 只用前 20w 条样本训练就足够了，不用跑全量
                break
                
    # 5. 开始训练
    print("🚀 Training Tokenizer (this may take 2-3 minutes)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # 6. 保存 (保存为 HF 格式)
    print(f"💾 Saving to {save_path}...")
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    
    # 为了让 AutoTokenizer 能直接加载，我们需要补充 config
    from transformers import PreTrainedTokenizerFast
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        unk_token="<|endoftext|>"
    )
    fast_tokenizer.save_pretrained(save_path)
    
    print("✅ Done! You can now load it via AutoTokenizer.from_pretrained('./tokenizer/minipile_32k')")

if __name__ == "__main__":
    train_custom_tokenizer()