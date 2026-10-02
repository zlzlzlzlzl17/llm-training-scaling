from collections import Counter
from hashlib import blake2b
from os import PathLike
from pathlib import Path
import itertools
import shutil
import unicodedata
from collections import defaultdict

import mmh3

def _hash_line(line: bytes) -> bytes:
    """Return a fixed-size hash for a line."""
    return blake2b(line, digest_size=16).digest()


def exact_line_deduplication(
    input_files: list[PathLike],
    output_directory: PathLike,
) -> None:
    """Remove every line that appears more than once across the corpus.

    Each output file has the same basename as its corresponding input file.
    Files are processed in binary mode so their original bytes and line
    endings are preserved.
    """
    input_paths = [Path(path) for path in input_files]
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid silently overwriting outputs when two inputs share a basename.
    filenames = [path.name for path in input_paths]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Input files must have unique filenames")

    # Pass 1: count every line across all input files.
    line_counts: Counter[bytes] = Counter()

    for input_path in input_paths:
        with input_path.open("rb") as input_file:
            for line in input_file:
                line_counts[_hash_line(line)] += 1

    # Pass 2: retain only lines occurring exactly once in the corpus.
    for input_path in input_paths:
        output_path = output_dir / input_path.name

        with (
            input_path.open("rb") as input_file,
            output_path.open("wb") as output_file,
        ):
            for line in input_file:
                if line_counts[_hash_line(line)] == 1:
                    output_file.write(line)


def _normalize_for_minhash(text: str) -> str:
    """Normalize text before constructing word n-grams."""
    # Decompose accented characters, e.g. "é" -> "e" + accent.
    text = unicodedata.normalize("NFD", text.lower())

    normalized_characters: list[str] = []

    for character in text:
        category = unicodedata.category(character)

        # Remove combining accents.
        if category.startswith("M"):
            continue

        # Replace punctuation with spaces so adjacent words do not merge.
        if category.startswith("P"):
            normalized_characters.append(" ")
        else:
            normalized_characters.append(character)

    # Collapse repeated whitespace and newlines.
    return " ".join("".join(normalized_characters).split())


def _get_word_ngrams(text: str, ngram_size: int) -> set[str]:
    """Convert normalized text into a set of word n-grams."""
    if ngram_size <= 0:
        raise ValueError("ngram_size must be positive")

    words = text.split()

    if not words:
        return {""}

    # Avoid giving all documents shorter than ngram_size an empty set.
    if len(words) < ngram_size:
        return {" ".join(words)}

    return {
        "\x1f".join(words[index : index + ngram_size])
        for index in range(len(words) - ngram_size + 1)
    }


def _compute_minhash_signature(
    document_ngrams: set[str],
    num_hashes: int,
) -> tuple[int, ...]:
    """Compute a MinHash signature using seeded MurmurHash functions."""
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")

    maximum_hash = (1 << 64) - 1
    signature: list[int] = []

    for seed in range(num_hashes):
        minimum_hash = maximum_hash

        for ngram in document_ngrams:
            hash_value = mmh3.hash64(
                ngram,
                seed=seed,
                signed=False,
            )[0]

            if hash_value < minimum_hash:
                minimum_hash = hash_value

        signature.append(minimum_hash)

    return tuple(signature)


def _jaccard_similarity(
    first: set[str],
    second: set[str],
) -> float:
    """Compute exact Jaccard similarity between two n-gram sets."""
    union = first | second

    if not union:
        return 1.0

    return len(first & second) / len(union)


class _UnionFind:
    """Disjoint-set structure for joining duplicate document clusters."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]

        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)

        if first_root == second_root:
            return

        # Keep the lower document index as the deterministic representative.
        if first_root < second_root:
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def minhash_deduplication(
    input_files: list[PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: PathLike,
) -> None:
    """Remove approximately duplicate documents using MinHash and LSH."""
    if num_hashes <= 0:
        raise ValueError("num_hashes must be positive")

    if num_bands <= 0:
        raise ValueError("num_bands must be positive")

    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be divisible by num_bands")

    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be between 0 and 1")

    # Sorting makes the retained representative deterministic.
    input_paths = sorted(
        (Path(path) for path in input_files),
        key=lambda path: path.name,
    )

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in input_paths
    ]

    document_ngram_sets = [
        _get_word_ngrams(
            _normalize_for_minhash(document),
            ngram_size=ngrams,
        )
        for document in documents
    ]

    signatures = [
        _compute_minhash_signature(
            document_ngrams=document_ngrams,
            num_hashes=num_hashes,
        )
        for document_ngrams in document_ngram_sets
    ]

    rows_per_band = num_hashes // num_bands

    # A bucket key includes the band index because equal values in different
    # bands do not constitute an LSH match.
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for document_index, signature in enumerate(signatures):
        for band_index in range(num_bands):
            start = band_index * rows_per_band
            end = start + rows_per_band
            band = signature[start:end]

            buckets[(band_index, band)].append(document_index)

    # Collect every document pair that shares at least one LSH bucket.
    candidate_pairs: set[tuple[int, int]] = set()

    for bucket_documents in buckets.values():
        if len(bucket_documents) < 2:
            continue

        candidate_pairs.update(
            itertools.combinations(bucket_documents, 2)
        )

    duplicate_clusters = _UnionFind(len(input_paths))

    # LSH only proposes candidates. Confirm duplication using true Jaccard.
    for first_index, second_index in candidate_pairs:
        similarity = _jaccard_similarity(
            document_ngram_sets[first_index],
            document_ngram_sets[second_index],
        )

        if similarity >= jaccard_threshold:
            duplicate_clusters.union(first_index, second_index)

    clusters: dict[int, list[int]] = defaultdict(list)

    for document_index in range(len(input_paths)):
        root = duplicate_clusters.find(document_index)
        clusters[root].append(document_index)

    # Keep the first document in each connected duplicate cluster.
    retained_indices = {
        min(cluster_members)
        for cluster_members in clusters.values()
    }

    for document_index, input_path in enumerate(input_paths):
        if document_index not in retained_indices:
            continue

        output_path = output_dir / input_path.name
        shutil.copyfile(input_path, output_path)