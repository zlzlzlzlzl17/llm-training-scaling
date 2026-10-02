import argparse
import shutil
from pathlib import Path

from llm_training_scaling.data_pipeline.deduplication import (
    exact_line_deduplication,
    minhash_deduplication,
)


def nonempty_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*.txt")
        if path.is_file() and path.stat().st_size > 0
    )


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate filtered text documents."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/filtered_1wet/documents"),
    )
    parser.add_argument(
        "--exact-dir",
        type=Path,
        default=Path("runs/filtered_1wet/exact_dedup"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/filtered_1wet/final_documents"),
    )
    parser.add_argument(
        "--num-hashes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--ngrams",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
    )

    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.txt"))

    if not input_files:
        raise FileNotFoundError(
            f"No .txt documents found in {args.input_dir}"
        )

    if args.num_hashes % args.num_bands != 0:
        raise ValueError(
            "num_hashes must be divisible by num_bands"
        )

    reset_directory(args.exact_dir)
    reset_directory(args.output_dir)

    print(f"Input documents: {len(input_files)}")

    print("\nRunning exact line deduplication...")
    exact_line_deduplication(
        input_files=input_files,
        output_directory=args.exact_dir,
    )

    exact_all_files = sorted(args.exact_dir.glob("*.txt"))
    exact_nonempty_files = nonempty_files(args.exact_dir)
    empty_after_exact = (
        len(exact_all_files) - len(exact_nonempty_files)
    )

    print(f"Exact-dedup outputs: {len(exact_all_files)}")
    print(f"Empty after exact dedup: {empty_after_exact}")
    print(
        "Non-empty after exact dedup: "
        f"{len(exact_nonempty_files)}"
    )

    if not exact_nonempty_files:
        raise RuntimeError(
            "All documents became empty after exact deduplication."
        )

    print("\nRunning MinHash + LSH deduplication...")
    minhash_deduplication(
        input_files=exact_nonempty_files,
        num_hashes=args.num_hashes,
        num_bands=args.num_bands,
        ngrams=args.ngrams,
        jaccard_threshold=args.jaccard_threshold,
        output_directory=args.output_dir,
    )

    final_all_files = sorted(args.output_dir.glob("*.txt"))
    final_nonempty_files = nonempty_files(args.output_dir)

    print("\nDeduplication complete")
    print("=" * 50)
    print(f"Original documents:       {len(input_files)}")
    print(f"After exact dedup:         {len(exact_nonempty_files)}")
    print(f"Final output files:        {len(final_all_files)}")
    print(f"Final non-empty documents: {len(final_nonempty_files)}")
    print(
        "Removed by MinHash:       "
        f"{len(exact_nonempty_files) - len(final_nonempty_files)}"
    )
    print("=" * 50)
    print(f"Final dataset: {args.output_dir}")


if __name__ == "__main__":
    main()