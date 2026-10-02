from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "HuggingFaceFW/dclm_100BT-shuffled"
TOKENIZER_NAME = "openlm-research/open_llama_3b_v2"


class TokenFileWriter:
    def __init__(
        self,
        path: Path,
        target_tokens: int,
        flush_tokens: int = 1_000_000,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")

        self.path = path
        self.target_tokens = target_tokens
        self.flush_tokens = flush_tokens
        self.count = 0
        self.buffer: list[int] = []

        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("wb")

    @property
    def complete(self) -> bool:
        return self.count >= self.target_tokens

    def add(self, token_ids: Iterable[int]) -> int:
        if self.complete:
            return 0

        remaining = self.target_tokens - self.count
        added = 0

        for token_id in token_ids:
            if added >= remaining:
                break

            if token_id < 0 or token_id > 65_535:
                raise ValueError(
                    f"Token ID cannot fit in uint16: {token_id}"
                )

            self.buffer.append(int(token_id))
            self.count += 1
            added += 1

            if len(self.buffer) >= self.flush_tokens:
                self.flush()

        return added

    def flush(self) -> None:
        if not self.buffer:
            return

        array = np.asarray(self.buffer, dtype="<u2")
        array.tofile(self.file)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.file.close()

        expected_size = self.count * 2
        actual_size = self.path.stat().st_size

        if actual_size != expected_size:
            raise RuntimeError(
                f"Incorrect file size for {self.path}: "
                f"expected {expected_size}, got {actual_size}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=3_000_000_000,
    )
    parser.add_argument(
        "--validation-tokens",
        type=int,
        default=2**18,
    )
    parser.add_argument(
        "--batch-documents",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_tokens_uint16.bin"
    validation_path = output_dir / "validation_tokens_uint16.bin"
    manifest_path = output_dir / "manifest.json"
    tokenizer_dir = output_dir / "tokenizer"

    existing_paths = [
        train_path,
        validation_path,
        manifest_path,
    ]

    if not args.overwrite:
        for path in existing_paths:
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists; pass --overwrite "
                    "to replace it"
                )

    print("Loading tokenizer:", TOKENIZER_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

    tokenizer.model_max_length = 10**12

    vocabulary_size = len(tokenizer)

    if vocabulary_size > 65_536:
        raise ValueError(
            f"Vocabulary is too large for uint16: {vocabulary_size}"
        )

    eos_token_id = tokenizer.eos_token_id

    if eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token")

    tokenizer.save_pretrained(tokenizer_dir)

    print("Tokenizer vocabulary:", vocabulary_size)
    print("EOS token ID:", eos_token_id)
    print("Loading streaming dataset:", DATASET_NAME)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    validation_writer = TokenFileWriter(
        validation_path,
        args.validation_tokens,
    )
    train_writer = TokenFileWriter(
        train_path,
        args.train_tokens,
    )

    documents_read = 0
    documents_used_for_validation = 0
    documents_used_for_training = 0

    document_batch: list[str] = []
    start_time = time.perf_counter()
    next_progress = args.progress_interval

    def process_batch(texts: list[str]) -> None:
        nonlocal documents_used_for_validation
        nonlocal documents_used_for_training
        nonlocal next_progress

        encoded_batch = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        for token_ids in encoded_batch:
            document_tokens = list(token_ids)
            document_tokens.append(eos_token_id)

            if not validation_writer.complete:
                validation_writer.add(document_tokens)
                documents_used_for_validation += 1

                # 即使最后一篇验证文档有剩余 token，
                # 也全部丢弃，避免同一文档进入训练集。
                continue

            if not train_writer.complete:
                train_writer.add(document_tokens)
                documents_used_for_training += 1

            if train_writer.count >= next_progress:
                elapsed = time.perf_counter() - start_time
                speed = (
                    train_writer.count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"Train tokens: {train_writer.count:,} / "
                    f"{args.train_tokens:,} | "
                    f"{speed:,.0f} tokens/s",
                    flush=True,
                )

                while next_progress <= train_writer.count:
                    next_progress += args.progress_interval

            if train_writer.complete:
                break

    try:
        for sample in dataset:
            text = sample.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            document_batch.append(text)
            documents_read += 1

            if len(document_batch) >= args.batch_documents:
                process_batch(document_batch)
                document_batch.clear()

            if train_writer.complete:
                break

        if document_batch and not train_writer.complete:
            process_batch(document_batch)
            document_batch.clear()

    finally:
        validation_writer.close()
        train_writer.close()

    if not validation_writer.complete:
        raise RuntimeError(
            "Dataset ended before validation target was reached"
        )

    if not train_writer.complete:
        raise RuntimeError(
            "Dataset ended before training target was reached"
        )

    elapsed_seconds = time.perf_counter() - start_time

    manifest = {
        "dataset": DATASET_NAME,
        "tokenizer": TOKENIZER_NAME,
        "vocabulary_size": vocabulary_size,
        "eos_token_id": eos_token_id,
        "dtype": "uint16-little-endian",
        "train_file": train_path.name,
        "validation_file": validation_path.name,
        "train_tokens": train_writer.count,
        "validation_tokens": validation_writer.count,
        "documents_read": documents_read,
        "documents_used_for_validation": (
            documents_used_for_validation
        ),
        "documents_used_for_training": (
            documents_used_for_training
        ),
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": (
            train_writer.count / elapsed_seconds
        ),
        "data_order": (
            "Sequential order from the globally shuffled "
            "streaming dataset"
        ),
    }

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("DCLM corpus completed")
    print("=" * 72)
    print(f"Train tokens:      {train_writer.count:,}")
    print(f"Validation tokens: {validation_writer.count:,}")
    print(
        f"Train file size:   "
        f"{train_path.stat().st_size / 1024**3:.3f} GiB"
    )
    print(
        f"Elapsed:           "
        f"{elapsed_seconds / 3600:.2f} hours"
    )
    print(
        f"Encoding speed:    "
        f"{manifest['tokens_per_second']:,.0f} tokens/s"
    )


    # Hugging Face streaming may leave a background network thread
    # active after the requested token prefix has been consumed.
    # At this point all files have been closed and the manifest has
    # already been written, so bypass interpreter finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
