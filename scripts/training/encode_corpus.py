from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from llm_training_scaling.model.tokenizer import Tokenizer


UINT16_MAX = np.iinfo(np.uint16).max


def encode_file(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    write_chunk_size: int,
) -> dict[str, Any]:
    """
    流式编码一个文本文件，并保存为连续 uint16 token IDs。

    输出文件是 raw binary，可以通过下面的方法读取：

        tokens = np.memmap(
            output_path,
            dtype=np.uint16,
            mode="r",
        )

    write_chunk_size 表示累计多少个 token ID 后写入一次磁盘。
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    if write_chunk_size <= 0:
        raise ValueError(
            "write_chunk_size must be positive."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_size_bytes = input_path.stat().st_size

    token_buffer: list[int] = []
    token_count = 0
    maximum_token_id = -1
    minimum_token_id = UINT16_MAX

    start_time = time.perf_counter()

    print()
    print("=" * 72)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Input size: "
        f"{input_size_bytes / 1024**3:.3f} GiB"
    )
    print(
        f"Write chunk size: "
        f"{write_chunk_size:,} tokens"
    )
    print("=" * 72)

    with (
        input_path.open(
            mode="r",
            encoding="utf-8",
        ) as input_file,
        output_path.open(
            mode="wb",
        ) as output_file,
    ):
        # input_file 本身是字符串 iterable。
        # encode_iterable() 会逐步读取，而不是一次读取整个文件。
        for token_id in tokenizer.encode_iterable(input_file):
            if token_id < 0 or token_id > UINT16_MAX:
                raise ValueError(
                    f"Token ID {token_id} cannot be represented "
                    "using uint16."
                )

            token_buffer.append(token_id)
            token_count += 1

            if token_id > maximum_token_id:
                maximum_token_id = token_id

            if token_id < minimum_token_id:
                minimum_token_id = token_id

            if len(token_buffer) >= write_chunk_size:
                np.asarray(
                    token_buffer,
                    dtype=np.uint16,
                ).tofile(output_file)

                token_buffer.clear()

                elapsed = time.perf_counter() - start_time
                tokens_per_second = (
                    token_count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    "\r"
                    f"Encoded {token_count:,} tokens | "
                    f"{tokens_per_second:,.0f} tokens/s",
                    end="",
                    flush=True,
                )

        # 写入最后一个不足 write_chunk_size 的 buffer。
        if token_buffer:
            np.asarray(
                token_buffer,
                dtype=np.uint16,
            ).tofile(output_file)

            token_buffer.clear()

    elapsed_seconds = time.perf_counter() - start_time
    output_size_bytes = output_path.stat().st_size

    expected_output_size = (
        token_count
        * np.dtype(np.uint16).itemsize
    )

    if output_size_bytes != expected_output_size:
        raise RuntimeError(
            "Output file size is incorrect. "
            f"Expected {expected_output_size:,} bytes, "
            f"got {output_size_bytes:,} bytes."
        )

    tokens_per_second = (
        token_count / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    input_bytes_per_second = (
        input_size_bytes / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    bytes_per_token = (
        input_size_bytes / token_count
        if token_count > 0
        else 0.0
    )

    print()
    print("-" * 72)
    print(
        f"Finished in:      "
        f"{elapsed_seconds:.2f} seconds"
    )
    print(
        f"Token count:      "
        f"{token_count:,}"
    )
    print(
        f"Minimum token ID: "
        f"{minimum_token_id}"
    )
    print(
        f"Maximum token ID: "
        f"{maximum_token_id}"
    )
    print(
        f"Tokens/second:    "
        f"{tokens_per_second:,.0f}"
    )
    print(
        f"Input speed:      "
        f"{input_bytes_per_second / 1024**2:.2f} MiB/s"
    )
    print(
        f"Bytes/token:      "
        f"{bytes_per_token:.3f}"
    )
    print(
        f"Output size:      "
        f"{output_size_bytes / 1024**2:.2f} MiB"
    )
    print("-" * 72)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dtype": "uint16",
        "input_size_bytes": input_size_bytes,
        "output_size_bytes": output_size_bytes,
        "token_count": token_count,
        "minimum_token_id": minimum_token_id,
        "maximum_token_id": maximum_token_id,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": tokens_per_second,
        "input_bytes_per_second": input_bytes_per_second,
        "bytes_per_token": bytes_per_token,
    }


def verify_output(
    tokenizer: Tokenizer,
    output_path: Path,
    sample_token_count: int = 200,
) -> None:
    """
    检查输出文件，并解码前面一小段 token。
    """

    tokens = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="r",
    )

    if len(tokens) == 0:
        raise RuntimeError(
            f"No tokens were written to {output_path}"
        )

    sample_size = min(
        sample_token_count,
        len(tokens),
    )

    sample_ids = (
        tokens[:sample_size]
        .astype(np.int64)
        .tolist()
    )

    decoded_sample = tokenizer.decode(sample_ids)

    print()
    print("Verification")
    print("-" * 72)
    print(
        f"Memmap token count: "
        f"{len(tokens):,}"
    )
    print(
        f"Dtype:              "
        f"{tokens.dtype}"
    )
    print(
        f"First 20 token IDs: "
        f"{sample_ids[:20]}"
    )
    print("Decoded sample:")
    print(repr(decoded_sample[:500]))
    print("-" * 72)
    print("Verification passed.")


def load_existing_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """
    读取已有 metadata。

    这样先运行 --only valid，再运行 --only train 时，
    不会删除之前保存的 valid metadata。
    """

    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (json.JSONDecodeError, OSError):
        pass

    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode TinyStories datasets into raw uint16 "
            "token-ID files."
        )
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(
            "artifacts/tinystories_bpe_10k/vocab.json"
        ),
        help="Path to vocab.json.",
    )

    parser.add_argument(
        "--merges",
        type=Path,
        default=Path(
            "artifacts/tinystories_bpe_10k/merges.json"
        ),
        help="Path to merges.json.",
    )

    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path(
            "data/TinyStoriesV2-GPT4-train.txt"
        ),
    )

    parser.add_argument(
        "--valid-input",
        type=Path,
        default=Path(
            "data/TinyStoriesV2-GPT4-valid.txt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/tinystories_tokens"
        ),
    )

    parser.add_argument(
        "--only",
        choices=(
            "train",
            "valid",
            "both",
        ),
        default="both",
        help="Choose which dataset split to encode.",
    )

    parser.add_argument(
        "--write-chunk-size",
        type=int,
        default=1_000_000,
        help=(
            "Number of token IDs buffered before each "
            "disk write."
        ),
    )

    parser.add_argument(
        "--sample-token-count",
        type=int,
        default=200,
        help=(
            "Number of initial tokens decoded during "
            "verification."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.vocab.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {args.vocab}"
        )

    if not args.merges.exists():
        raise FileNotFoundError(
            f"Merges file not found: {args.merges}"
        )

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )

    print(
        f"Loaded tokenizer with "
        f"{len(tokenizer.vocab):,} tokens."
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        args.output_dir
        / "tokenization_metadata.json"
    )

    metadata = load_existing_metadata(
        metadata_path
    )

    metadata["tokenizer"] = {
        "vocab_path": str(args.vocab),
        "merges_path": str(args.merges),
        "vocab_size": len(tokenizer.vocab),
        "special_tokens": ["<|endoftext|>"],
    }

    if args.only in {"valid", "both"}:
        valid_output_path = (
            args.output_dir
            / "tinystories_valid_tokens_uint16.bin"
        )

        metadata["valid"] = encode_file(
            tokenizer=tokenizer,
            input_path=args.valid_input,
            output_path=valid_output_path,
            write_chunk_size=args.write_chunk_size,
        )

        verify_output(
            tokenizer=tokenizer,
            output_path=valid_output_path,
            sample_token_count=args.sample_token_count,
        )

    if args.only in {"train", "both"}:
        train_output_path = (
            args.output_dir
            / "tinystories_train_tokens_uint16.bin"
        )

        metadata["train"] = encode_file(
            tokenizer=tokenizer,
            input_path=args.train_input,
            output_path=train_output_path,
            write_chunk_size=args.write_chunk_size,
        )

        verify_output(
            tokenizer=tokenizer,
            output_path=train_output_path,
            sample_token_count=args.sample_token_count,
        )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("Encoding completed successfully.")
    print(f"Metadata: {metadata_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
