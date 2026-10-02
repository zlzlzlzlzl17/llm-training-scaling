import argparse
from pathlib import Path

from llm_training_scaling.data_pipeline.wet_files import EnglishWetFiles


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-files",
        type=int,
        default=64,
        help="Number of original Common Crawl WET files.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of original WET files combined into each output chunk.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/aistation_64wet/wet_paths.txt"),
    )

    args = parser.parse_args()

    if args.n_files <= 0:
        raise ValueError("n_files must be positive")

    if args.group_size <= 0:
        raise ValueError("group_size must be positive")

    if args.n_files % args.group_size != 0:
        raise ValueError("n_files must be divisible by group_size")

    dataset = EnglishWetFiles(
        n_files=args.n_files,
        group_size=args.group_size,
    )

    paths = dataset.load_or_create()

    args.manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.manifest.open("w", encoding="utf-8") as file:
        for path in paths:
            file.write(str(path) + "\n")

    print()
    print("=" * 60)
    print(f"Original WET files requested: {args.n_files}")
    print(f"Files per output chunk:       {args.group_size}")
    print(f"Generated WET chunks:         {len(paths)}")
    print(f"Manifest:                     {args.manifest}")
    print("=" * 60)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
