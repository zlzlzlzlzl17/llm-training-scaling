from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--category",
        choices=["format_1_answer_0", "format_0_answer_0"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    shown = 0

    with args.path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row["category"] != args.category:
                continue

            print("=" * 100)
            print("INDEX:", row["index"])
            print("QUESTION:")
            print(row["question"])
            print("\nGROUND TRUTH:")
            print(row["ground_truth"])
            print("\nRESPONSE:")
            print(row["response"])
            print()

            shown += 1
            if shown >= args.limit:
                break

    print(f"Displayed {shown} examples.")


if __name__ == "__main__":
    main()
