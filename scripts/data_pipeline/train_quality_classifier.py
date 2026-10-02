import argparse
import gzip
import random
import re
import urllib.request
from pathlib import Path

import fasttext
from warcio.archiveiterator import ArchiveIterator

from llm_training_scaling.data_pipeline.extraction import extract_text_from_html_bytes
from llm_training_scaling.data_pipeline.language import identify_language


USER_AGENT = (
    "cs336-assignment4-data/0.1 "
    "(educational use; "
    "https://github.com/stanford-cs336/assignment4-data)"
)


def normalize_fasttext_text(text: str) -> str:
    """Convert one document into one fastText training line."""
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_usable_english(text: str) -> bool:
    """Apply lightweight filtering shared by both classes."""
    if len(text) < 300:
        return False

    language, confidence = identify_language(text)
    return language == "en" and confidence >= 0.7


def load_wiki_urls(path: Path, seed: int) -> list[str]:
    with gzip.open(path, "rt", errors="ignore") as file:
        urls = {
            line.strip()
            for line in file
            if line.startswith(("http://", "https://"))
        }

    urls = list(urls)
    random.Random(seed).shuffle(urls)
    return urls


def fetch_wiki_examples(
    urls: list[str],
    target_count: int,
    timeout: float,
) -> list[str]:
    examples: list[str] = []

    # Do not attempt an unlimited number of broken or inaccessible URLs.
    maximum_attempts = max(target_count * 20, 1000)

    for attempt_number, url in enumerate(urls[:maximum_attempts], start=1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )

            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                content_type = response.headers.get(
                    "Content-Type",
                    "",
                ).lower()

                if (
                    "text/html" not in content_type
                    and "application/xhtml+xml" not in content_type
                ):
                    continue

                # Prevent extremely large pages from consuming excessive memory.
                html_bytes = response.read(5_000_000)

            text = extract_text_from_html_bytes(html_bytes)
            text = normalize_fasttext_text(text)

            if not is_usable_english(text):
                continue

            examples.append(text)

            if len(examples) % 50 == 0:
                print(
                    f"[wiki] collected {len(examples)}/{target_count} "
                    f"after {attempt_number} attempts",
                    flush=True,
                )

            if len(examples) >= target_count:
                break

        except Exception:
            # Many old Wikipedia references are offline, blocked,
            # redirected incorrectly, or too slow.
            continue

    return examples


def find_wet_files(root: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in root.rglob("*.warc.wet.gz")
        if path.is_file()
    )

    if not candidates:
        candidates = sorted(
            path
            for path in root.rglob("*.wet.gz")
            if path.is_file()
        )

    return candidates


def extract_cc_examples(
    wet_paths: list[Path],
    target_count: int,
    seed: int,
) -> list[str]:
    examples: list[str] = []

    # Reservoir sampling gives a random sample without storing every document.
    rng = random.Random(seed)
    usable_documents_seen = 0

    for wet_path in wet_paths:
        print(f"[cc] reading {wet_path}", flush=True)

        with gzip.open(wet_path, "rb") as input_stream:
            for record in ArchiveIterator(input_stream):
                if record.rec_type != "conversion":
                    continue

                payload = record.content_stream().read()
                text = payload.decode(
                    "utf-8",
                    errors="replace",
                )
                text = normalize_fasttext_text(text)

                if not is_usable_english(text):
                    continue

                usable_documents_seen += 1

                if len(examples) < target_count:
                    examples.append(text)
                else:
                    replacement_index = rng.randrange(
                        usable_documents_seen
                    )

                    if replacement_index < target_count:
                        examples[replacement_index] = text

    rng.shuffle(examples)
    return examples


def write_fasttext_dataset(
    path: Path,
    examples: list[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for label, text in examples:
            file.write(f"__label__{label} {text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("local-shared-data"),
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=336,
    )

    args = parser.parse_args()

    wiki_url_path = (
        args.data_root
        / "wiki"
        / "enwiki-20260501-extracted_urls.txt.gz"
    )

    if not wiki_url_path.exists():
        raise FileNotFoundError(
            f"Wikipedia URL file not found: {wiki_url_path}"
        )

    wet_paths = find_wet_files(args.data_root)

    if not wet_paths:
        raise FileNotFoundError(
            f"No WET files found below {args.data_root}"
        )

    print(f"[wiki] loading URLs from {wiki_url_path}", flush=True)
    wiki_urls = load_wiki_urls(
        wiki_url_path,
        seed=args.seed,
    )

    print(f"[wiki] found {len(wiki_urls)} unique URLs", flush=True)

    wiki_examples = fetch_wiki_examples(
        urls=wiki_urls,
        target_count=args.target_per_class,
        timeout=args.timeout,
    )

    print(
        f"[wiki] final usable examples: {len(wiki_examples)}",
        flush=True,
    )

    # Balance the CC class against however many wiki pages were retrieved.
    balanced_target = min(
        args.target_per_class,
        len(wiki_examples),
    )

    if balanced_target < 20:
        raise RuntimeError(
            "Too few Wikipedia examples were downloaded. "
            "Try processing more Wikipedia shards."
        )

    wiki_examples = wiki_examples[:balanced_target]

    cc_examples = extract_cc_examples(
        wet_paths=wet_paths,
        target_count=balanced_target,
        seed=args.seed,
    )

    balanced_target = min(
        len(wiki_examples),
        len(cc_examples),
    )

    wiki_examples = wiki_examples[:balanced_target]
    cc_examples = cc_examples[:balanced_target]

    print(
        f"[dataset] balanced examples per class: {balanced_target}",
        flush=True,
    )

    rng = random.Random(args.seed)

    wiki_labeled = [
        ("wiki", text)
        for text in wiki_examples
    ]
    cc_labeled = [
        ("cc", text)
        for text in cc_examples
    ]

    rng.shuffle(wiki_labeled)
    rng.shuffle(cc_labeled)

    validation_count = max(
        1,
        int(
            balanced_target
            * args.validation_fraction
        ),
    )

    validation_examples = (
        wiki_labeled[:validation_count]
        + cc_labeled[:validation_count]
    )

    training_examples = (
        wiki_labeled[validation_count:]
        + cc_labeled[validation_count:]
    )

    rng.shuffle(training_examples)
    rng.shuffle(validation_examples)

    dataset_dir = args.data_root / "quality-classifier"
    train_path = dataset_dir / "train.txt"
    validation_path = dataset_dir / "validation.txt"

    write_fasttext_dataset(
        train_path,
        training_examples,
    )
    write_fasttext_dataset(
        validation_path,
        validation_examples,
    )

    print(
        f"[dataset] training examples: {len(training_examples)}",
        flush=True,
    )
    print(
        f"[dataset] validation examples: {len(validation_examples)}",
        flush=True,
    )

    print("[fasttext] training classifier", flush=True)

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=15,
        lr=0.5,
        wordNgrams=2,
        dim=100,
        minCount=2,
        loss="softmax",
        thread=4,
        verbose=2,
    )

    classifier_path = (
        args.data_root
        / "classifiers"
        / "quality_classifier.bin"
    )
    classifier_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(str(classifier_path))

    sample_count, precision, recall = model.test(
        str(validation_path)
    )

    print(f"[fasttext] saved model: {classifier_path}")
    print(f"[fasttext] validation samples: {sample_count}")
    print(f"[fasttext] precision@1: {precision:.4f}")
    print(f"[fasttext] recall@1: {recall:.4f}")


if __name__ == "__main__":
    main()