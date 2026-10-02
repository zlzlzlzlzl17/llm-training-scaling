from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from llm_training_scaling.model.transformer import TransformerLM, softmax
from llm_training_scaling.model.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained TinyStories Transformer."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "runs/tinystories_15m/checkpoints/final.pt"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "runs/tinystories_15m/config.json"
        ),
    )

    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(
            "artifacts/tinystories_bpe_10k/vocab.json"
        ),
    )

    parser.add_argument(
        "--merges",
        type=Path,
        default=Path(
            "artifacts/tinystories_bpe_10k/merges.json"
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time,",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "0 means greedy decoding. "
            "Values around 0.7–1.0 are typical."
        ),
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/tinystories_15m/generated.txt"
        ),
    )

    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is unavailable."
        )

    return device


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    return checkpoint


def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """
    Nucleus sampling。

    保留按概率从高到低累计达到 top_p 所需的最小 token 集合，
    其他 token 的 logit 设置为 -inf。
    """

    if top_p >= 1.0:
        return logits

    if top_p <= 0.0:
        raise ValueError(
            f"top_p must be in (0, 1], got {top_p}"
        )

    sorted_logits, sorted_indices = torch.sort(
        logits,
        dim=-1,
        descending=True,
    )

    sorted_probabilities = softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    # 第一个使累计概率超过 top_p 的 token 本身仍然保留，
    # 所以将删除 mask 向右移动一位。
    remove_mask = cumulative_probabilities > top_p

    remove_mask[..., 1:] = remove_mask[
        ..., :-1
    ].clone()

    remove_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove_mask,
        float("-inf"),
    )

    filtered_logits = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """
    logits shape:
        (batch_size, vocab_size)

    return shape:
        (batch_size, 1)
    """

    if temperature < 0:
        raise ValueError(
            "temperature cannot be negative."
        )

    # temperature=0：greedy decoding。
    if temperature == 0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    scaled_logits = logits / temperature

    filtered_logits = apply_top_p(
        scaled_logits,
        top_p=top_p,
    )

    probabilities = softmax(
        filtered_logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )


def make_autocast_context(
    device: torch.device,
):
    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        return torch.autocast(
            device_type="cuda",
            dtype=dtype,
        )

    return nullcontext()


@torch.inference_mode()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[str, int, bool]:
    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError(
            "The prompt must encode to at least one token."
        )

    token_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    end_token_id = tokenizer.special_token_to_id[
        "<|endoftext|>"
    ]

    generated_count = 0
    stopped_on_eos = False

    for _ in range(max_new_tokens):
        # 模型最多只能接收 context_length 个 token。
        model_input = token_ids[
            :, -model.context_length:
        ]

        with make_autocast_context(device):
            logits = model(model_input)

        # 只使用最后一个位置对下一个 token 的预测。
        next_token_logits = logits[
            :, -1, :
        ].float()

        next_token = sample_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_p=top_p,
        )

        next_token_id = int(next_token.item())

        if next_token_id == end_token_id:
            stopped_on_eos = True
            break

        token_ids = torch.cat(
            (token_ids, next_token),
            dim=1,
        )

        generated_count += 1

    full_ids = token_ids[0].tolist()
    generated_text = tokenizer.decode(full_ids)

    return generated_text, generated_count, stopped_on_eos


def main() -> None:
    args = parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be positive."
        )

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    config = load_config(args.config)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=torch.float32,
    )

    checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    iteration = checkpoint.get(
        "iteration",
        "unknown",
    )

    print("=" * 72)
    print("TinyStories generation")
    print("=" * 72)
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Training step:   {iteration}")
    print(f"Device:          {device}")
    print(f"Prompt:          {args.prompt!r}")
    print(f"Temperature:     {args.temperature}")
    print(f"Top-p:           {args.top_p}")
    print(f"Maximum tokens:  {args.max_new_tokens}")
    print("=" * 72)

    start_time = time.perf_counter()

    text, generated_count, stopped_on_eos = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    elapsed = time.perf_counter() - start_time

    print()
    print(text)
    print()
    print("=" * 72)
    print(f"Generated tokens: {generated_count}")
    print(f"Stopped on EOS:   {stopped_on_eos}")
    print(f"Elapsed seconds:  {elapsed:.2f}")

    if elapsed > 0:
        print(
            f"Tokens/second:   "
            f"{generated_count / elapsed:.2f}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(text)
        file.write("\n")

    print(f"Saved output:     {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
