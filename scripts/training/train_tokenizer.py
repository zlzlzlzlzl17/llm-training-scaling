from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from llm_training_scaling.model.bpe import train_bpe
from llm_training_scaling.model.tokenizer import Tokenizer


def get_peak_rss_mb() -> float:
    """
    返回当前进程的 peak resident-set size。

    Linux/WSL 上 ru_maxrss 的单位是 KiB；
    macOS 上单位是 bytes。
    """

    peak_rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":
        return peak_rss / (1024 * 1024)

    return peak_rss / 1024


def save_vocab(
    vocab: dict[int, bytes],
    output_path: Path,
) -> None:
    """
    将 vocabulary 保存为 JSON。

    bytes 不能直接写入 JSON，因此使用 hexadecimal string。

    例如：
        b"a"   -> "61"
        b"the" -> "746865"
    """

    serialized_vocab = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in sorted(
            vocab.items()
        )
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialized_vocab,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_merges(
    merges: list[tuple[bytes, bytes]],
    output_path: Path,
) -> None:
    """
    将 merges 保存为 JSON-hex 格式。
    """

    serialized_merges = [
        [
            left.hex(),
            right.hex(),
        ]
        for left, right in merges
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialized_merges,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_readable_vocab(
    vocab: dict[int, bytes],
    output_path: Path,
) -> None:
    """
    额外保存人类可读版本，方便检查 vocabulary。
    """

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "token_id\tbyte_length\tbytes_repr\tdecoded_text\n"
        )

        for token_id, token_bytes in sorted(
            vocab.items()
        ):
            decoded_text = token_bytes.decode(
                "utf-8",
                errors="replace",
            )

            # 避免真正的换行符和 tab 破坏文件格式。
            escaped_text = (
                decoded_text
                .replace("\\", "\\\\")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )

            file.write(
                f"{token_id}\t"
                f"{len(token_bytes)}\t"
                f"{token_bytes!r}\t"
                f"{escaped_text}\n"
            )


def save_readable_merges(
    merges: list[tuple[bytes, bytes]],
    output_path: Path,
) -> None:
    """
    保存可读 merge 顺序。
    """

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "rank\tleft\tright\tmerged\n"
        )

        for rank, (left, right) in enumerate(merges):
            merged = left + right

            file.write(
                f"{rank}\t"
                f"{left!r}\t"
                f"{right!r}\t"
                f"{merged!r}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a byte-level BPE tokenizer on TinyStories."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/TinyStoriesV2-GPT4-train.txt"
        ),
        help="Path to the TinyStories training text.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help=(
            "Maximum final vocabulary size, including "
            "byte tokens and special tokens."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/tinystories_bpe_10k"
        ),
        help="Directory in which to save tokenizer files.",
    )

    parser.add_argument(
        "--special-token",
        dest="special_tokens",
        action="append",
        default=None,
        help=(
            "Special token. May be supplied more than once. "
            "Default: <|endoftext|>"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_dir: Path = args.output_dir
    vocab_size: int = args.vocab_size

    special_tokens: list[str] = (
        args.special_tokens
        if args.special_tokens is not None
        else ["<|endoftext|>"]
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Training file does not exist: {input_path}"
        )

    if vocab_size <= 0:
        raise ValueError(
            "vocab_size must be positive."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("TinyStories BPE training")
    print("=" * 70)
    print(f"Input:          {input_path}")
    print(f"Input size:     {input_path.stat().st_size / 1024**2:.2f} MiB")
    print(f"Vocab size:     {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print(f"Output dir:     {output_dir}")
    print()

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    peak_rss_mb = get_peak_rss_mb()

    longest_token_id, longest_token = max(
        vocab.items(),
        key=lambda item: len(item[1]),
    )

    longest_token_text = longest_token.decode(
        "utf-8",
        errors="replace",
    )

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.json"
    metadata_path = output_dir / "metadata.json"
    readable_vocab_path = output_dir / "vocab_readable.txt"
    readable_merges_path = output_dir / "merges_readable.txt"

    save_vocab(
        vocab=vocab,
        output_path=vocab_path,
    )

    save_merges(
        merges=merges,
        output_path=merges_path,
    )

    save_readable_vocab(
        vocab=vocab,
        output_path=readable_vocab_path,
    )

    save_readable_merges(
        merges=merges,
        output_path=readable_merges_path,
    )

    metadata = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": len(vocab),
        "merge_count": len(merges),
        "special_tokens": special_tokens,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_mb": peak_rss_mb,
        "longest_token": {
            "token_id": longest_token_id,
            "byte_length": len(longest_token),
            "hex": longest_token.hex(),
            "bytes_repr": repr(longest_token),
            "decoded_text": longest_token_text,
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------------------------------------------
    # 加载刚刚保存的文件，验证 serialization 和 tokenizer。
    # ----------------------------------------------------------

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

    test_text = (
        "Once upon a time, there was a little cat."
        "<|endoftext|>"
    )

    test_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(test_ids)

    if decoded_text != test_text:
        raise RuntimeError(
            "Tokenizer roundtrip check failed.\n"
            f"Expected: {test_text!r}\n"
            f"Actual:   {decoded_text!r}"
        )

    print()
    print("=" * 70)
    print("Training completed")
    print("=" * 70)
    print(f"Elapsed time:       {elapsed_seconds:.2f} seconds")
    print(f"Peak RSS:           {peak_rss_mb:.2f} MiB")
    print(f"Actual vocab size:  {len(vocab)}")
    print(f"Merge count:        {len(merges)}")
    print(f"Longest token ID:   {longest_token_id}")
    print(f"Longest byte length:{len(longest_token)}")
    print(f"Longest token bytes:{longest_token!r}")
    print(f"Longest token text: {longest_token_text!r}")
    print()
    print("Saved files:")
    print(f"  {vocab_path}")
    print(f"  {merges_path}")
    print(f"  {metadata_path}")
    print(f"  {readable_vocab_path}")
    print(f"  {readable_merges_path}")
    print()
    print("Roundtrip check passed.")
    print(f"Example token IDs: {test_ids}")


if __name__ == "__main__":
    main()