import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from filter_single_wet_file import process_wet_file


def process_one(
    index: int,
    wet_path: str,
    output_root: str,
) -> tuple[int, str, dict[str, int]]:
    input_path = Path(wet_path)
    chunk_dir = Path(output_root) / f"chunk_{index:03d}"
    stats_path = chunk_dir / "stats.json"

    # Resume completed chunks.
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return index, str(input_path), stats

    # Remove an incomplete previous attempt.
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    stats = process_wet_file(
        input_path=input_path,
        output_dir=chunk_dir,
        max_documents=None,
    )

    return index, str(input_path), stats


def write_summary(
    completed: list[dict],
    output_root: Path,
) -> None:
    totals: dict[str, int] = {}

    for item in completed:
        for name, value in item["stats"].items():
            totals[name] = totals.get(name, 0) + int(value)

    summary = {
        "completed_chunks": len(completed),
        "totals": totals,
        "chunks": sorted(completed, key=lambda x: x["index"]),
    }

    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def combine_inspection_samples(output_root: Path) -> None:
    combined: list[dict] = []

    for chunk_dir in sorted(output_root.glob("chunk_*")):
        sample_path = chunk_dir / "inspection_samples.json"

        if not sample_path.exists():
            continue

        samples = json.loads(sample_path.read_text(encoding="utf-8"))

        for reason, items in samples.items():
            for item in items:
                combined.append(
                    {
                        "chunk": chunk_dir.name,
                        "reason": reason,
                        **item,
                    }
                )

    (output_root / "combined_inspection_samples.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter multiple English Common Crawl WET chunks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    wet_paths = [
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not wet_paths:
        raise RuntimeError("Manifest contains no WET paths.")

    missing = [path for path in wet_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing WET files:\n" + "\n".join(missing)
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    completed: list[dict] = []

    print(f"WET chunks: {len(wet_paths)}")
    print(f"Workers:    {args.workers}")
    print(f"Output:     {args.output_root}")
    print("=" * 60, flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one,
                index,
                wet_path,
                str(args.output_root),
            ): index
            for index, wet_path in enumerate(wet_paths)
        }

        for future in as_completed(futures):
            index, wet_path, stats = future.result()

            completed.append(
                {
                    "index": index,
                    "wet_path": wet_path,
                    "stats": stats,
                }
            )

            write_summary(completed, args.output_root)

            print(
                f"[{len(completed):02d}/{len(wet_paths):02d}] "
                f"chunk_{index:03d}: "
                f"processed={stats['conversion_records']} "
                f"kept={stats['kept']}",
                flush=True,
            )

    write_summary(completed, args.output_root)
    combine_inspection_samples(args.output_root)

    totals = {
        name: sum(
            int(item["stats"].get(name, 0))
            for item in completed
        )
        for name in completed[0]["stats"]
    }

    print("\nBatch filtering complete")
    print("=" * 60)

    for name, value in totals.items():
        print(f"{name:24s} {value}")

    print("=" * 60)
    print(f"Summary: {args.output_root / 'summary.json'}")
    print(
        "Samples: "
        f"{args.output_root / 'combined_inspection_samples.json'}"
    )


if __name__ == "__main__":
    main()
