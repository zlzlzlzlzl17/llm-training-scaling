from __future__ import annotations

import argparse
import csv
import gc
from collections.abc import Callable
from pathlib import Path

import torch
import triton

from llm_training_scaling.systems.attention import pytorch_attention
from llm_training_scaling.systems.flash_attention_triton import FlashAttentionTriton


DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

PROVIDER_MAP = {
    "pytorch": pytorch_attention,
    "triton": FlashAttentionTriton.apply,
}


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def bench(
    fn: Callable[[], object],
    warmup_ms: int,
    rep_ms: int,
) -> float:
    result = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
    )
    return float(result)


def run_forward(
    attention_fn: Callable[..., torch.Tensor],
    sequence_length: int,
    d: int,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    q = torch.randn(
        1,
        sequence_length,
        d,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    try:
        def forward() -> torch.Tensor:
            return attention_fn(q, k, v, True)

        return bench(forward, warmup_ms, rep_ms)
    finally:
        del q, k, v
        cleanup()


def run_backward(
    attention_fn: Callable[..., torch.Tensor],
    sequence_length: int,
    d: int,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    q = torch.randn(
        1,
        sequence_length,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    output = attention_fn(q, k, v, True)
    grad_output = torch.randn_like(output)

    # Reuse the same forward graph so only backward is timed.
    def backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            outputs=output,
            inputs=(q, k, v),
            grad_outputs=grad_output,
            retain_graph=True,
        )

    try:
        return bench(backward, warmup_ms, rep_ms)
    finally:
        del q, k, v, output, grad_output
        cleanup()


def run_end_to_end(
    attention_fn: Callable[..., torch.Tensor],
    sequence_length: int,
    d: int,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> float:
    q = torch.randn(
        1,
        sequence_length,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    grad_output = torch.randn_like(q)

    def forward_backward() -> tuple[torch.Tensor, ...]:
        output = attention_fn(q, k, v, True)

        return torch.autograd.grad(
            outputs=output,
            inputs=(q, k, v),
            grad_outputs=grad_output,
        )

    try:
        return bench(
            forward_backward,
            warmup_ms,
            rep_ms,
        )
    finally:
        del q, k, v, grad_output
        cleanup()


def is_oom(exc: BaseException) -> bool:
    return (
        isinstance(exc, torch.OutOfMemoryError)
        or "out of memory" in str(exc).lower()
    )


def run_metric(
    metric_fn: Callable[[], float],
) -> tuple[float | str, str, str]:
    try:
        value = metric_fn()
        return value, "ok", ""
    except Exception as exc:
        cleanup()

        status = "oom" if is_oom(exc) else "error"
        error = str(exc).replace("\n", " ")[:1000]

        return "", status, error


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "provider",
        "dtype",
        "batch_size",
        "sequence_length",
        "d",
        "is_causal",
        "forward_ms",
        "forward_status",
        "backward_ms",
        "backward_status",
        "end_to_end_ms",
        "end_to_end_status",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-lengths",
        type=parse_int_list,
        default=[
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
    )
    parser.add_argument(
        "--dimensions",
        type=parse_int_list,
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--dtypes",
        type=parse_str_list,
        default=["float32", "bfloat16"],
    )
    parser.add_argument(
        "--providers",
        type=parse_str_list,
        default=["pytorch", "triton"],
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "../results/flash_attention/"
            "flash_attention_benchmark.csv"
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    # Let PyTorch FP32 matmul use Tensor Cores/TF32 on A800,
    # making comparison with Triton tl.dot fairer.
    torch.set_float32_matmul_precision("high")

    rows: list[dict[str, object]] = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print()

    for dtype_name in args.dtypes:
        dtype = DTYPE_MAP[dtype_name]

        for d in args.dimensions:
            for sequence_length in args.sequence_lengths:
                for provider_name in args.providers:
                    attention_fn = PROVIDER_MAP[provider_name]

                    print(
                        f"provider={provider_name:7s} "
                        f"dtype={dtype_name:8s} "
                        f"seq={sequence_length:6d} "
                        f"d={d:3d}"
                    )

                    forward_ms, forward_status, forward_error = (
                        run_metric(
                            lambda: run_forward(
                                attention_fn,
                                sequence_length,
                                d,
                                dtype,
                                args.warmup_ms,
                                args.rep_ms,
                            )
                        )
                    )

                    backward_ms, backward_status, backward_error = (
                        run_metric(
                            lambda: run_backward(
                                attention_fn,
                                sequence_length,
                                d,
                                dtype,
                                args.warmup_ms,
                                args.rep_ms,
                            )
                        )
                    )

                    end_to_end_ms, end_status, end_error = (
                        run_metric(
                            lambda: run_end_to_end(
                                attention_fn,
                                sequence_length,
                                d,
                                dtype,
                                args.warmup_ms,
                                args.rep_ms,
                            )
                        )
                    )

                    errors = [
                        x
                        for x in (
                            forward_error,
                            backward_error,
                            end_error,
                        )
                        if x
                    ]

                    row = {
                        "provider": provider_name,
                        "dtype": dtype_name,
                        "batch_size": 1,
                        "sequence_length": sequence_length,
                        "d": d,
                        "is_causal": True,
                        "forward_ms": forward_ms,
                        "forward_status": forward_status,
                        "backward_ms": backward_ms,
                        "backward_status": backward_status,
                        "end_to_end_ms": end_to_end_ms,
                        "end_to_end_status": end_status,
                        "error": " | ".join(errors),
                    }

                    rows.append(row)
                    write_csv(args.output, rows)

                    print(
                        f"  forward={forward_ms} "
                        f"({forward_status}), "
                        f"backward={backward_ms} "
                        f"({backward_status}), "
                        f"e2e={end_to_end_ms} "
                        f"({end_status})"
                    )

    write_csv(args.output, rows)
    print()
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
