import argparse
import gzip
import json
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator

from llm_training_scaling.data_pipeline.harmful import classify_nsfw, classify_toxic_speech
from llm_training_scaling.data_pipeline.pii import mask_emails, mask_ips, mask_phone_numbers
from llm_training_scaling.data_pipeline.quality import classify_quality, gopher_quality_filter


def save_sample(
    samples: dict[str, list[dict[str, str | float]]],
    reason: str,
    *,
    url: str,
    text: str,
    label: str = "",
    score: float = 0.0,
    max_samples: int = 5,
) -> None:
    """Save a small number of examples for manual inspection."""
    bucket = samples.setdefault(reason, [])

    if len(bucket) >= max_samples:
        return

    bucket.append(
        {
            "url": url,
            "label": label,
            "score": float(score),
            "excerpt": text[:1000],
        }
    )


def process_wet_file(
    input_path: Path,
    output_dir: Path,
    max_documents: int | None,
) -> dict[str, int]:
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "warc_records": 0,
        "conversion_records": 0,
        "empty": 0,
        "removed_gopher": 0,
        "removed_quality": 0,
        "removed_nsfw": 0,
        "removed_toxic": 0,
        "emails_masked": 0,
        "phones_masked": 0,
        "ips_masked": 0,
        "kept": 0,
    }

    samples: dict[str, list[dict[str, str | float]]] = {}

    with gzip.open(input_path, "rb") as stream:
        for record in ArchiveIterator(stream):
            stats["warc_records"] += 1

            if record.rec_type != "conversion":
                continue

            if (
                max_documents is not None
                and stats["conversion_records"] >= max_documents
            ):
                break

            stats["conversion_records"] += 1

            url = (
                record.rec_headers.get_header("WARC-Target-URI")
                or ""
            )

            payload = record.content_stream().read()
            text = payload.decode("utf-8", errors="replace")
            text = text.replace("\x00", "").strip()

            if not text:
                stats["empty"] += 1
                save_sample(
                    samples,
                    "empty",
                    url=url,
                    text=text,
                )
                continue

            # Cheap rule-based filtering comes first.
            if not gopher_quality_filter(text):
                stats["removed_gopher"] += 1
                save_sample(
                    samples,
                    "gopher",
                    url=url,
                    text=text,
                )
                continue

            # Keep only pages judged Wikipedia-like by the trained classifier.
            quality_label, quality_score = classify_quality(text)

            if quality_label != "wiki":
                stats["removed_quality"] += 1
                save_sample(
                    samples,
                    "quality",
                    url=url,
                    text=text,
                    label=quality_label,
                    score=quality_score,
                )
                continue

            nsfw_label, nsfw_score = classify_nsfw(text)

            if nsfw_label == "nsfw":
                stats["removed_nsfw"] += 1
                save_sample(
                    samples,
                    "nsfw",
                    url=url,
                    text=text,
                    label=nsfw_label,
                    score=nsfw_score,
                )
                continue

            toxic_label, toxic_score = classify_toxic_speech(text)

            if toxic_label == "toxic":
                stats["removed_toxic"] += 1
                save_sample(
                    samples,
                    "toxic",
                    url=url,
                    text=text,
                    label=toxic_label,
                    score=toxic_score,
                )
                continue

            # Mask PII only after the document has passed all filters.
            text, email_count = mask_emails(text)
            text, phone_count = mask_phone_numbers(text)
            text, ip_count = mask_ips(text)

            stats["emails_masked"] += email_count
            stats["phones_masked"] += phone_count
            stats["ips_masked"] += ip_count

            output_path = (
                documents_dir
                / f"{stats['kept']:08d}.txt"
            )

            output_path.write_text(
                text.rstrip() + "\n",
                encoding="utf-8",
            )

            stats["kept"] += 1

            if stats["conversion_records"] % 1000 == 0:
                print(
                    f"processed={stats['conversion_records']} "
                    f"kept={stats['kept']}",
                    flush=True,
                )

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "stats.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            stats,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "inspection_samples.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            samples,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter an English Common Crawl WET file."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an English-filtered .warc.wet.gz file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/filtered_1wet"),
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional limit for a small test run.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    stats = process_wet_file(
        input_path=args.input,
        output_dir=args.output_dir,
        max_documents=args.max_documents,
    )

    print("\nFiltering complete")
    print("=" * 50)

    for name, value in stats.items():
        print(f"{name:24s} {value}")

    print("=" * 50)
    print(f"Documents: {args.output_dir / 'documents'}")
    print(f"Statistics: {args.output_dir / 'stats.json'}")
    print(
        "Inspection samples: "
        f"{args.output_dir / 'inspection_samples.json'}"
    )


if __name__ == "__main__":
    main()