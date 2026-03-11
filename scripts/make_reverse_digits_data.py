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
PATTERN_ID = 20
DELAY_ID = 21
REPEAT_ID = 22
RUNMAX_ID = 23
THRESH_ID = 24
BRACKET_ID = 25
MASK_ID = 26
LPAREN_ID = 27
RPAREN_ID = 28

IGNORE_INDEX = -100

OPERATION_ID_BY_NAME = {
    "reverse_digits": 0,
    "copy_digits": 1,
    "parity_digits": 2,
    "cumsum_mod10": 3,
    "majority_vote": 4,
    "count_ones": 5,
    "pattern_complete": 6,
    "delayed_copy": 7,
    "first_repeat": 8,
    "running_max": 9,
    "sum_threshold": 10,
    "bracket_depth": 11,
}

CONTROL_TOKEN_BY_OPERATION = {
    "reverse_digits": REV_ID,
    "copy_digits": COPY_ID,
    "parity_digits": PARITY_ID,
    "cumsum_mod10": CUMSUM_ID,
    "majority_vote": MAJORITY_ID,
    "count_ones": COUNT_ID,
    "pattern_complete": PATTERN_ID,
    "delayed_copy": DELAY_ID,
    "first_repeat": REPEAT_ID,
    "running_max": RUNMAX_ID,
    "sum_threshold": THRESH_ID,
    "bracket_depth": BRACKET_ID,
}


def encode_digit(value: int) -> int:
    return DIGIT_OFFSET + int(value)


def encode_decimal_number(value: int) -> list[int]:
    return [encode_digit(int(ch)) for ch in str(int(value))]


def _make_row(
    *,
    prefix_tokens: list[int],
    output_tokens: list[int],
    length: int,
    task_name: str,
    operation_name: str,
) -> dict[str, list[int] | int | str]:
    sequence = [BOS_ID, CONTROL_TOKEN_BY_OPERATION[operation_name]] + prefix_tokens + [SEP_ID] + output_tokens + [EOS_ID]
    input_ids = sequence[:-1]
    targets = sequence[1:]
    supervised_mask = [0] * (len(prefix_tokens) + 2) + [1] * (len(output_tokens) + 1)
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


def _random_digits(length: int, rng: random.Random, upper: int = 10) -> list[int]:
    return [rng.randrange(upper) for _ in range(length)]


def build_reverse_example(length: int, rng: random.Random, task_name: str = "reverse_digits") -> dict[str, list[int] | int | str]:
    digits = _random_digits(length, rng)
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(value) for value in reversed(digits)],
        length=length,
        task_name=task_name,
        operation_name="reverse_digits",
    )


def build_copy_example(length: int, rng: random.Random, task_name: str = "copy_digits") -> dict[str, list[int] | int | str]:
    digits = _random_digits(length, rng)
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(value) for value in digits],
        length=length,
        task_name=task_name,
        operation_name="copy_digits",
    )


def build_parity_example(length: int, rng: random.Random, task_name: str = "parity_digits") -> dict[str, list[int] | int | str]:
    digits = _random_digits(length, rng)
    answer = sum(value % 2 for value in digits) % 2
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(answer)],
        length=length,
        task_name=task_name,
        operation_name="parity_digits",
    )


def build_cumsum_mod10_example(length: int, rng: random.Random, task_name: str = "cumsum_mod10") -> dict[str, list[int] | int | str]:
    digits = _random_digits(length, rng)
    running = 0
    outputs = []
    for value in digits:
        running = (running + value) % 10
        outputs.append(encode_digit(running))
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=outputs,
        length=length,
        task_name=task_name,
        operation_name="cumsum_mod10",
    )


def _binary_digits(length: int, rng: random.Random) -> list[int]:
    return _random_digits(length, rng, upper=2)


def build_majority_vote_example(length: int, rng: random.Random, task_name: str = "majority_vote") -> dict[str, list[int] | int | str]:
    digits = _binary_digits(length, rng)
    ones = sum(digits)
    zeros = length - ones
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(int(ones > zeros))],
        length=length,
        task_name=task_name,
        operation_name="majority_vote",
    )


def build_count_ones_example(length: int, rng: random.Random, task_name: str = "count_ones") -> dict[str, list[int] | int | str]:
    digits = _binary_digits(length, rng)
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=encode_decimal_number(sum(digits)),
        length=length,
        task_name=task_name,
        operation_name="count_ones",
    )


def build_pattern_complete_example(
    length: int,
    rng: random.Random,
    task_name: str = "pattern_complete",
) -> dict[str, list[int] | int | str]:
    """
    Pattern completion with a single masked token.

    The underlying pattern is a simple cyclic progression over digits. The
    missing token is a pure completion target, which is friendly to exact-match
    symbolic retrieval.
    """
    start = rng.randrange(10)
    digits = [(start + offset) % 10 for offset in range(length)]
    mask_pos = rng.randint(1, length - 2)
    answer = digits[mask_pos]
    prefix = [encode_digit(value) for value in digits]
    prefix[mask_pos] = MASK_ID
    return _make_row(
        prefix_tokens=prefix,
        output_tokens=[encode_digit(answer)],
        length=length,
        task_name=task_name,
        operation_name="pattern_complete",
    )


def build_delayed_copy_example(
    length: int,
    rng: random.Random,
    task_name: str = "delayed_copy",
) -> dict[str, list[int] | int | str]:
    """
    Delayed copy with a fixed delay proportional to the sequence length.

    This rewards stateful retention of earlier symbols rather than local suffix
    matching.
    """
    digits = _random_digits(length, rng)
    delay = max(1, length // 2)
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(value) for value in digits[:-delay]],
        length=length,
        task_name=task_name,
        operation_name="delayed_copy",
    )


def build_first_repeat_example(
    length: int,
    rng: random.Random,
    task_name: str = "first_repeat",
) -> dict[str, list[int] | int | str]:
    """
    Output the first value whose second occurrence appears in the sequence.
    """
    seen: set[int] = set()
    digits: list[int] = []
    first_repeat: int | None = None

    while len(digits) < length:
        value = rng.randrange(10)
        digits.append(value)
        if value in seen and first_repeat is None:
            first_repeat = value
        seen.add(value)

    if first_repeat is None:
        pos = rng.randint(length // 2, length - 1)
        digits[pos] = digits[0]
        first_repeat = digits[0]

    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(first_repeat)],
        length=length,
        task_name=task_name,
        operation_name="first_repeat",
    )


def build_running_max_example(
    length: int,
    rng: random.Random,
    task_name: str = "running_max",
) -> dict[str, list[int] | int | str]:
    digits = _random_digits(length, rng)
    current_max = 0
    outputs = []
    for value in digits:
        current_max = max(current_max, value)
        outputs.append(encode_digit(current_max))
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=outputs,
        length=length,
        task_name=task_name,
        operation_name="running_max",
    )


def build_sum_threshold_example(
    length: int,
    rng: random.Random,
    task_name: str = "sum_threshold",
) -> dict[str, list[int] | int | str]:
    """
    Smooth threshold task.

    For length 20 this reduces to the intuitive "sum > 100" rule. For general
    lengths we scale the threshold linearly as ``5 * length`` so the label stays
    balanced across train and OOD lengths.
    """
    digits = _random_digits(length, rng)
    threshold = 5 * length
    answer = int(sum(digits) > threshold)
    return _make_row(
        prefix_tokens=[encode_digit(value) for value in digits],
        output_tokens=[encode_digit(answer)],
        length=length,
        task_name=task_name,
        operation_name="sum_threshold",
    )


def build_bracket_depth_example(
    length: int,
    rng: random.Random,
    task_name: str = "bracket_depth",
) -> dict[str, list[int] | int | str]:
    """
    Output the maximum bracket nesting depth as a decimal number.
    """
    depth = 0
    max_depth = 0
    prefix: list[int] = []
    for pos in range(length):
        remaining = length - pos
        if depth <= 0:
            choose_open = True
        elif depth >= remaining:
            choose_open = False
        else:
            choose_open = rng.random() < 0.55

        if choose_open:
            depth += 1
            prefix.append(LPAREN_ID)
        else:
            depth -= 1
            prefix.append(RPAREN_ID)
        max_depth = max(max_depth, depth)

    return _make_row(
        prefix_tokens=prefix,
        output_tokens=encode_decimal_number(max_depth),
        length=length,
        task_name=task_name,
        operation_name="bracket_depth",
    )


TASK_BUILDERS = {
    "reverse_digits": build_reverse_example,
    "copy_digits": build_copy_example,
    "parity_digits": build_parity_example,
    "cumsum_mod10": build_cumsum_mod10_example,
    "majority_vote": build_majority_vote_example,
    "count_ones": build_count_ones_example,
    "pattern_complete": build_pattern_complete_example,
    "delayed_copy": build_delayed_copy_example,
    "first_repeat": build_first_repeat_example,
    "running_max": build_running_max_example,
    "sum_threshold": build_sum_threshold_example,
    "bracket_depth": build_bracket_depth_example,
    "mixed_digits": None,
    "mixed_all_digits": None,
    "mixed_v3": None,
}


def _build_round_robin_mix(
    *,
    size: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
    task_name: str,
    operation_names: list[str],
) -> list[dict[str, list[int] | int | str]]:
    rows = []
    for idx in range(size):
        op_name = operation_names[idx % len(operation_names)]
        builder = TASK_BUILDERS[op_name]
        rows.append(builder(rng.randint(min_len, max_len), rng, task_name=task_name))
    rng.shuffle(rows)
    return rows


def build_split(task: str, size: int, min_len: int, max_len: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    if task == "mixed_digits":
        rows = _build_round_robin_mix(
            size=size,
            min_len=min_len,
            max_len=max_len,
            rng=rng,
            task_name="mixed_digits",
            operation_names=["reverse_digits", "copy_digits"],
        )
    elif task == "mixed_v3":
        rows = _build_round_robin_mix(
            size=size,
            min_len=min_len,
            max_len=max_len,
            rng=rng,
            task_name="mixed_v3",
            operation_names=[
                "reverse_digits",
                "copy_digits",
                "pattern_complete",
                "delayed_copy",
                "first_repeat",
                "running_max",
                "sum_threshold",
                "bracket_depth",
            ],
        )
    elif task == "mixed_all_digits":
        rows = _build_round_robin_mix(
            size=size,
            min_len=min_len,
            max_len=max_len,
            rng=rng,
            task_name="mixed_all_digits",
            operation_names=[
                "reverse_digits",
                "copy_digits",
                "parity_digits",
                "cumsum_mod10",
                "majority_vote",
                "count_ones",
                "pattern_complete",
                "delayed_copy",
                "first_repeat",
                "running_max",
                "sum_threshold",
                "bracket_depth",
            ],
        )
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
