"""Generate small sequence-to-sequence toy datasets for ROSA/CaMoE experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from datasets import Dataset, DatasetDict

PAD_ID = 0
BOS_ID = 1
SEP_ID = 2
EOS_ID = 3
DIGIT_OFFSET = 4
REV_ID = 14
COPY_ID = 15
PARITY_ID = 16
CUMSUM_ID = 17
MAJORITY_ID = 18
COUNT_ID = 19
IGNORE_INDEX = -100
OPERATION_ID_BY_NAME = {
    "reverse_digits": 0,
    "copy_digits": 1,
    "parity_digits": 2,
    "cumsum_mod10": 3,
    "majority_vote": 4,
    "count_ones": 5,
}
CONTROL_TOKEN_BY_OPERATION = {
    "reverse_digits": REV_ID,
    "copy_digits": COPY_ID,
    "parity_digits": PARITY_ID,
    "cumsum_mod10": CUMSUM_ID,
    "majority_vote": MAJORITY_ID,
    "count_ones": COUNT_ID,
}


def encode_decimal_number(value: int) -> list[int]:
    return [DIGIT_OFFSET + int(ch) for ch in str(int(value))]


def _build_example(
    *,
    length: int,
    rng: random.Random,
    task_name: str,
    operation_name: str,
    output_builder: callable,
) -> dict[str, list[int] | int | str]:
    digits = [rng.randrange(10) for _ in range(length)]
    encoded_digits = [DIGIT_OFFSET + value for value in digits]
    output_tokens = output_builder(digits, encoded_digits)
    sequence = [BOS_ID, CONTROL_TOKEN_BY_OPERATION[operation_name]] + encoded_digits + [SEP_ID] + output_tokens + [EOS_ID]
    input_ids = sequence[:-1]
    targets = sequence[1:]
    supervised_mask = [0] * (length + 2) + [1] * (len(output_tokens) + 1)
    masked_targets = [target if mask else IGNORE_INDEX for target, mask in zip(targets, supervised_mask)]
    return {
        "input_ids": input_ids,
        "targets": masked_targets,
        "supervised_mask": supervised_mask,
        "length": length,
        "task": task_name,
        "operation": operation_name,
        "operation_id": OPERATION_ID_BY_NAME[operation_name],
    }


def build_reverse_example(length: int, rng: random.Random, task_name: str = "reverse_digits") -> dict[str, list[int] | int | str]:
    return _build_example(
        length=length,
        rng=rng,
        task_name=task_name,
        operation_name="reverse_digits",
        output_builder=lambda _digits, values: list(reversed(values)),
    )


def build_copy_example(length: int, rng: random.Random, task_name: str = "copy_digits") -> dict[str, list[int] | int | str]:
    return _build_example(
        length=length,
        rng=rng,
        task_name=task_name,
        operation_name="copy_digits",
        output_builder=lambda _digits, values: list(values),
    )


def build_parity_example(length: int, rng: random.Random, task_name: str = "parity_digits") -> dict[str, list[int] | int | str]:
    return _build_example(
        length=length,
        rng=rng,
        task_name=task_name,
        operation_name="parity_digits",
        output_builder=lambda digits, _values: [DIGIT_OFFSET + (sum(value % 2 for value in digits) % 2)],
    )


def build_cumsum_mod10_example(
    length: int,
    rng: random.Random,
    task_name: str = "cumsum_mod10",
) -> dict[str, list[int] | int | str]:
    def build_outputs(digits: list[int], _values: list[int]) -> list[int]:
        running = 0
        outputs = []
        for value in digits:
            running = (running + value) % 10
            outputs.append(DIGIT_OFFSET + running)
        return outputs

    return _build_example(
        length=length,
        rng=rng,
        task_name=task_name,
        operation_name="cumsum_mod10",
        output_builder=build_outputs,
    )


def build_majority_vote_example(
    length: int,
    rng: random.Random,
    task_name: str = "majority_vote",
) -> dict[str, list[int] | int | str]:
    def build_outputs(digits: list[int], _values: list[int]) -> list[int]:
        ones = sum(digits)
        zeros = len(digits) - ones
        return [DIGIT_OFFSET + int(ones > zeros)]

    return _build_binary_example(
        length=length,
        rng=rng,
        task_name=task_name,
        operation_name="majority_vote",
        output_builder=build_outputs,
    )


def build_count_ones_example(
    length: int,
    rng: random.Random,
    task_name: str = "count_ones",
) -> dict[str, list[int] | int | str]:
    def build_outputs(digits: list[int], _values: list[int]) -> list[int]:
        return encode_decimal_number(sum(digits))

    return _build_binary_example(
        length=length,
        rng=rng,
        task_name=task_name,
        operation_name="count_ones",
        output_builder=build_outputs,
    )


def _build_binary_example(
    *,
    length: int,
    rng: random.Random,
    task_name: str,
    operation_name: str,
    output_builder: callable,
) -> dict[str, list[int] | int | str]:
    digits = [rng.randrange(2) for _ in range(length)]
    encoded_digits = [DIGIT_OFFSET + value for value in digits]
    output_tokens = output_builder(digits, encoded_digits)
    sequence = [BOS_ID, CONTROL_TOKEN_BY_OPERATION[operation_name]] + encoded_digits + [SEP_ID] + output_tokens + [EOS_ID]
    input_ids = sequence[:-1]
    targets = sequence[1:]
    supervised_mask = [0] * (length + 2) + [1] * (len(output_tokens) + 1)
    masked_targets = [target if mask else IGNORE_INDEX for target, mask in zip(targets, supervised_mask)]
    return {
        "input_ids": input_ids,
        "targets": masked_targets,
        "supervised_mask": supervised_mask,
        "length": length,
        "task": task_name,
        "operation": operation_name,
        "operation_id": OPERATION_ID_BY_NAME[operation_name],
    }


TASK_BUILDERS = {
    "reverse_digits": build_reverse_example,
    "copy_digits": build_copy_example,
    "parity_digits": build_parity_example,
    "cumsum_mod10": build_cumsum_mod10_example,
    "majority_vote": build_majority_vote_example,
    "count_ones": build_count_ones_example,
    "mixed_digits": None,
    "mixed_all_digits": None,
}


def build_split(task: str, size: int, min_len: int, max_len: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    if task == "mixed_digits":
        reverse_count = size // 2
        copy_count = size - reverse_count
        rows = [
            build_reverse_example(rng.randint(min_len, max_len), rng, task_name="mixed_digits")
            for _ in range(reverse_count)
        ]
        rows.extend(
            build_copy_example(rng.randint(min_len, max_len), rng, task_name="mixed_digits")
            for _ in range(copy_count)
        )
        rng.shuffle(rows)
    elif task == "mixed_all_digits":
        builders = [
            lambda length: build_reverse_example(length, rng, task_name="mixed_all_digits"),
            lambda length: build_copy_example(length, rng, task_name="mixed_all_digits"),
            lambda length: build_parity_example(length, rng, task_name="mixed_all_digits"),
            lambda length: build_cumsum_mod10_example(length, rng, task_name="mixed_all_digits"),
            lambda length: build_majority_vote_example(length, rng, task_name="mixed_all_digits"),
            lambda length: build_count_ones_example(length, rng, task_name="mixed_all_digits"),
        ]
        rows = []
        for idx in range(size):
            builder = builders[idx % len(builders)]
            rows.append(builder(rng.randint(min_len, max_len)))
        rng.shuffle(rows)
    else:
        builder = TASK_BUILDERS[task]
        rows = [builder(rng.randint(min_len, max_len), rng) for _ in range(size)]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create toy sequence datasets")
    parser.add_argument("--task", type=str, choices=sorted(TASK_BUILDERS), default="reverse_digits")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--val_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=5000)
    parser.add_argument("--ood_size", type=int, default=5000)
    parser.add_argument("--train_min_len", type=int, default=4)
    parser.add_argument("--train_max_len", type=int, default=20)
    parser.add_argument("--ood_min_len", type=int, default=21)
    parser.add_argument("--ood_max_len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path("data") / args.task
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = DatasetDict(
        {
            "train": build_split(args.task, args.train_size, args.train_min_len, args.train_max_len, args.seed + 1),
            "validation": build_split(args.task, args.val_size, args.train_min_len, args.train_max_len, args.seed + 2),
            "test": build_split(args.task, args.test_size, args.train_min_len, args.train_max_len, args.seed + 3),
            "ood": build_split(args.task, args.ood_size, args.ood_min_len, args.ood_max_len, args.seed + 4),
        }
    )
    dataset.save_to_disk(str(output_dir))
    print(f"task={args.task}")
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
