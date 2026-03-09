"""lm-evaluation-harness entrypoint for CaMoE v22."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Iterable

import torch
import torch.nn.functional as F

import lm_eval
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from camoe.backbone import init_rwkv7_cuda
from camoe.model import load_camoe_checkpoint

try:
    import pyrwkv_tokenizer

    RUST_TOKENIZER = True
except ImportError:
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

    RUST_TOKENIZER = False


def chunks(items, size: int) -> Iterable[list]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def pad_to_chunk(input_ids: torch.Tensor, chunk_len: int = 16) -> tuple[torch.Tensor, int]:
    if input_ids.size(1) % chunk_len == 0:
        return input_ids, 0
    pad_len = chunk_len - (input_ids.size(1) % chunk_len)
    padding = torch.zeros((input_ids.size(0), pad_len), dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat([input_ids, padding], dim=1), pad_len


class CaMoELM(LM):
    def __init__(
        self,
        pretrained: str,
        scale: str,
        device: str,
        batch_size: int,
        vocab_file: str = "tokenizer/rwkv_vocab_v20230424.txt",
    ) -> None:
        super().__init__()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._batch_size = int(batch_size)
        del scale
        self.model, self.config, _ = load_camoe_checkpoint(pretrained, device=self._device, config=None, strict=True)
        self.model.eval()
        self._max_length = self.config.seq_len
        self._eot_token_id = 0
        self._pad_token_id = 0
        if RUST_TOKENIZER:
            self.tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
            self.is_rust_tokenizer = True
        else:
            self.tokenizer = TRIE_TOKENIZER(vocab_file)
            self.is_rust_tokenizer = False

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_length(self) -> int:
        return self._max_length

    def tok_encode(self, string: str, add_special_tokens: bool = False):
        del add_special_tokens
        return self.tokenizer.encode(string) if string else []

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        padded_ids, pad_len = pad_to_chunk(input_ids)
        with torch.no_grad():
            result = self.model(
                padded_ids,
                critic_alpha=1.0,
                training=False,
                uniform=False,
            )
        logits = result["logits"]
        if pad_len > 0:
            logits = logits[:, :-pad_len, :]
        return logits

    def loglikelihood(self, requests: list[Instance]):
        results = []
        for batch in chunks(requests, self.batch_size):
            batch_input_ids = []
            ctx_lens = []
            cont_lens = []
            for req in batch:
                context, continuation = req.args
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation) or [self.eot_token_id]
                full_ids = (ctx_ids + cont_ids)[-self.max_length :]
                ctx_len = max(0, len(full_ids) - len(cont_ids))
                batch_input_ids.append(full_ids)
                ctx_lens.append(ctx_len)
                cont_lens.append(len(cont_ids))

            max_len = max(len(ids) for ids in batch_input_ids)
            padded = [ids + [self._pad_token_id] * (max_len - len(ids)) for ids in batch_input_ids]
            input_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            logits = self._model_call(input_tensor)

            shift_logits = logits[:, :-1, :]
            shift_labels = input_tensor[:, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            greedy_tokens = shift_logits.argmax(dim=-1)
            is_correct = greedy_tokens == shift_labels

            mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
            for row, (ctx_len, cont_len) in enumerate(zip(ctx_lens, cont_lens)):
                start = max(ctx_len - 1, 0)
                end = min(start + cont_len, token_log_probs.size(1))
                if start < end:
                    mask[row, start:end] = True

            sum_log_probs = (token_log_probs * mask).sum(dim=1)
            all_correct = ((is_correct & mask).sum(dim=1) == mask.sum(dim=1))
            results.extend(zip(sum_log_probs.tolist(), all_correct.tolist()))
        return results

    def loglikelihood_rolling(self, requests: list[Instance]):
        results = []
        for batch in chunks(requests, self.batch_size):
            batch_input_ids = []
            valid_lens = []
            for req in batch:
                (text,) = req.args
                ids = self.tok_encode(text)[: self.max_length]
                batch_input_ids.append(ids)
                valid_lens.append(len(ids))

            max_len = max(len(ids) for ids in batch_input_ids)
            padded = [ids + [self._pad_token_id] * (max_len - len(ids)) for ids in batch_input_ids]
            input_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            logits = self._model_call(input_tensor)

            for row, valid_len in enumerate(valid_lens):
                if valid_len <= 1:
                    results.append((0.0, True))
                    continue
                sample_logits = logits[row, : valid_len - 1, :]
                sample_targets = input_tensor[row, 1:valid_len]
                log_probs = F.log_softmax(sample_logits, dim=-1)
                token_log_probs = log_probs.gather(1, sample_targets.unsqueeze(1)).squeeze(1)
                greedy = sample_logits.argmax(dim=-1)
                results.append((token_log_probs.sum().item(), bool((greedy == sample_targets).all().item())))
        return results

    def generate_until(self, requests: list[Instance]):
        results = []
        for req in requests:
            context = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            until = gen_kwargs.get("until", ["\n"])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))

            ids = self.tok_encode(context)[-self.max_length :]
            generated = ids[:]
            for _ in range(max_gen_toks):
                input_tensor = torch.tensor([generated[-self.max_length :]], dtype=torch.long, device=self.device)
                logits = self._model_call(input_tensor)
                next_token = int(logits[0, -1].argmax().item())
                generated.append(next_token)
                text = self.tok_decode(generated[len(ids) :])
                if any(stop in text for stop in until) or next_token == self.eot_token_id:
                    break
            final_text = self.tok_decode(generated[len(ids) :])
            for stop in until:
                if stop in final_text:
                    final_text = final_text.split(stop)[0]
                    break
            results.append(final_text)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness on CaMoE v22")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--scale", default="0.4b", choices=["0.1b", "0.4b"])
    parser.add_argument("--tasks", default="lambada_openai")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    init_rwkv7_cuda()
    lm = CaMoELM(
        pretrained=args.pretrained,
        scale=args.scale,
        device=args.device,
        batch_size=args.batch_size,
    )

    tasks_list = [task.strip() for task in args.tasks.split(",") if task.strip()]
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks_list,
        batch_size=args.batch_size,
    )

    if args.output:
        output_path = args.output
    else:
        task_slug = "_".join(tasks_list)[:64]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_v22_{args.scale}_{task_slug}_{timestamp}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print(json.dumps(results.get("results", results), indent=2, ensure_ascii=False))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
