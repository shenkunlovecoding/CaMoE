import argparse
from pathlib import Path
import sys


DEFAULT_FILES = [
    "train.py",
    "CaMoE/system.py",
    "CaMoE/market.py",
    "CaMoE/critic.py",
    "CaMoE/experts.py",
    "CaMoE/bridge.py",
    "CaMoE/backbone.py",
    "CaMoE/config.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump CaMoE core modules and train.py as plain text for AI context."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text file path. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Override default file list, e.g. --files train.py CaMoE/system.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    file_list = args.files if args.files else DEFAULT_FILES

    chunks = []
    for rel in file_list:
        path = repo_root / rel
        if not path.exists():
            chunks.append(f"{rel}:\n[FILE NOT FOUND]\n")
            continue

        content = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"{rel}:\n{content}\n")

    output_text = "\n".join(chunks)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"Saved to: {out_path}")
    else:
        sys.stdout.buffer.write(output_text.encode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
