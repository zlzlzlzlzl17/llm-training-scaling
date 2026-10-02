from __future__ import annotations

import argparse
import csv
import gc
import statistics
from pathlib import Path
from typing import Callable

import torch

from llm_training_scaling.systems.attention import pytorch_attention


DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_integer_list(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got {value!r}"
        ) from exc

    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "All values must be positive integers"
        )

    return result


def allocated_mib() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def reserved_mib() -> float:
    return torch.cuda.memory_reserved() / (1024**2)


def elapsed_ms(
    operation: Callable[[], None],
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    operation()
    end.record()

    end.synchronize()
    return start.elapsed_time(end)


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_case(
    sequence_length: int,
    d: int,
    dtype: torch.dtype,
    batch_size: int,
    warmup: int,
    repetitions: int,
    compiled: bool,
) -> dict[str, object]:
    device = torch.device("cuda")

    q = torch.randn(
        batch_size,
        sequence_length,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        sequence_length,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        sequence_length,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    attention_fn: Callable[..., torch.Tensor] = pytorch_attention

    if compiled:
        attention_fn = torch.compile(
            pytorch_attention,
            fullgraph=True,
        )

    # Warm-up includes compilation and initial CUDA kernel setup.
    for _ in range(warmup):
        output = attention_fn(q, k, v, False)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        q.grad = None
        k.grad = None
        v.grad = None

        del output, grad_output

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    forward_times: list[float] = []
    backward_times: list[float] = []
    memory_before_backward: list[float] = []

    for _ in range(repetitions):
        q.grad = None
        k.grad = None
        v.grad = None

        output_holder: list[torch.Tensor] = []

        def run_forward() -> None:
            output_holder.append(
                attention_fn(q, k, v, False)
            )

        forward_times.append(elapsed_ms(run_forward))

        output = output_holder.pop()
        memory_before_backward.append(allocated_mib())
        grad_output = torch.randn_like(output)

        def run_backward() -> None:
            output.backward(grad_output)

        backward_times.append(elapsed_ms(run_backward))

        del output, grad_output

    torch.cuda.synchronize()

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "d": d,
        "dtype": str(dtype).removeprefix("torch."),
        "compiled": compiled,
        "warmup": warmup,
        "repetitions": repetitions,
        "forward_mean_ms": statistics.mean(forward_times),
        "forward_std_ms": (
            statistics.stdev(forward_times)
            if len(forward_times) > 1
            else 0.0
        ),
        "backward_mean_ms": statistics.mean(backward_times),
        "backward_std_ms": (
            statistics.stdev(backward_times)
            if len(backward_times) > 1
            else 0.0
        ),
        "memory_before_backward_mib": max(
            memory_before_backward
        ),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
        ),
        "reserved_memory_mib": reserved_mib(),
        "status": "ok",
        "error": "",
    }


def write_results(
    output_path: Path,
    rows: list[dict[str, object]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark standard PyTorch attention."
    )
    parser.add_argument(
        "--sequence-lengths",
        type=parse_integer_list,
        default=[256, 1024, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--dimensions",
        type=parse_integer_list,
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP,
        default="float32",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--compiled",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "../results/attention/"
            "attention_eager_float32.csv"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")

    dtype = DTYPE_MAP[args.dtype]
    rows: list[dict[str, object]] = []

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype: {dtype}")
    print(f"compiled: {args.compiled}")
    print()

    for d in args.dimensions:
        for sequence_length in args.sequence_lengths:
            clear_cuda()

            print(
                f"Running batch={args.batch_size}, "
                f"seq={sequence_length}, d={d}, "
                f"dtype={args.dtype}, "
                f"compiled={args.compiled}"
            )

            try:
                result = benchmark_case(
                    sequence_length=sequence_length,
                    d=d,
                    dtype=dtype,
                    batch_size=args.batch_size,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                    compiled=args.compiled,
                )

            except torch.OutOfMemoryError as exc:
                result = {
                    "batch_size": args.batch_size,
                    "sequence_length": sequence_length,
                    "d": d,
                    "dtype": args.dtype,
                    "compiled": args.compiled,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "forward_mean_ms": "",
                    "forward_std_ms": "",
                    "backward_mean_ms": "",
                    "backward_std_ms": "",
                    "memory_before_backward_mib": "",
                    "peak_memory_mib": "",
                    "reserved_memory_mib": "",
                    "status": "oom",
                    "error": str(exc).replace("\n", " "),
                }

            except RuntimeError as exc:
                # Some CUDA OOMs are raised as a generic RuntimeError.
                if "out of memory" not in str(exc).lower():
                    raise

                result = {
                    "batch_size": args.batch_size,
                    "sequence_length": sequence_length,
                    "d": d,
                    "dtype": args.dtype,
                    "compiled": args.compiled,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "forward_mean_ms": "",
                    "forward_std_ms": "",
                    "backward_mean_ms": "",
                    "backward_std_ms": "",
                    "memory_before_backward_mib": "",
                    "peak_memory_mib": "",
                    "reserved_memory_mib": "",
                    "status": "oom",
                    "error": str(exc).replace("\n", " "),
                }

            rows.append(result)
            write_results(args.output, rows)

            if result["status"] == "ok":
                print(
                    f"  forward: "
                    f"{result['forward_mean_ms']:.3f} "
                    f"± {result['forward_std_ms']:.3f} ms"
                )
                print(
                    f"  backward: "
                    f"{result['backward_mean_ms']:.3f} "
                    f"± {result['backward_std_ms']:.3f} ms"
                )
                print(
                    f"  memory before backward: "
                    f"{result['memory_before_backward_mib']:.1f} MiB"
                )
                print(
                    f"  peak memory: "
                    f"{result['peak_memory_mib']:.1f} MiB"
                )
            else:
                print("  OOM")

            print()

            clear_cuda()

            if args.compiled:
                # Prevent compiled graphs for old shapes accumulating
                # across the entire parameter sweep.
                torch._dynamo.reset()

    write_results(args.output, rows)
    print(f"Saved results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
