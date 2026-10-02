import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


_TOKENIZER = None


def init_worker(tokenizer_name: str) -> None:
    global _TOKENIZER
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_document(path_string: str) -> list[int]:
    if _TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized.")

    path = Path(path_string)
    text = path.read_text(encoding="utf-8", errors="replace")

    token_ids = _TOKENIZER.encode(
        text,
        add_special_tokens=False,
    )
    token_ids.append(_TOKENIZER.eos_token_id)

    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize filtered documents using GPT-2 tokenizer."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
    )
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.txt"))

    if not input_files:
        raise FileNotFoundError(
            f"No .txt files found in {args.input_dir}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    document_count = 0

    context = mp.get_context("spawn")

    with args.output.open("wb") as output_file:
        with context.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args.tokenizer,),
        ) as pool:
            results = pool.imap(
                tokenize_document,
                (str(path) for path in input_files),
                chunksize=16,
            )

            for token_ids in tqdm(
                results,
                total=len(input_files),
                desc="Tokenizing",
            ):
                ids_array = np.asarray(
                    token_ids,
                    dtype=np.uint16,
                )
                ids_array.tofile(output_file)

                total_tokens += len(token_ids)
                document_count += 1

    stats = {
        "documents": document_count,
        "tokens": total_tokens,
        "dtype": "uint16",
        "tokenizer": args.tokenizer,
        "eos_token_id": 50256,
        "output_path": str(args.output),
        "output_bytes": args.output.stat().st_size,
    }

    stats_path = args.output.with_suffix(
        args.output.suffix + ".stats.json"
    )
    stats_path.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print()
    print("Tokenization complete")
    print("=" * 60)
    print(f"Documents:    {document_count:,}")
    print(f"Tokens:       {total_tokens:,}")
    print(f"Output size:  {args.output.stat().st_size / 1024**2:.2f} MiB")
    print(f"Output:       {args.output}")
    print(f"Statistics:   {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
