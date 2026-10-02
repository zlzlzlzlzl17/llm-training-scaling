from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes-mib",
        type=int,
        nargs="+",
        default=[1, 10, 100, 1024],
        help="Tensor sizes in MiB.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def setup() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return local_rank, rank, world_size


def benchmark_size(
    size_mib: int,
    local_rank: int,
    rank: int,
    world_size: int,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    device = torch.device(f"cuda:{local_rank}")

    size_bytes = size_mib * 1024 * 1024
    num_elements = size_bytes // torch.tensor(
        [],
        dtype=torch.float32,
    ).element_size()

    tensor = torch.empty(
        num_elements,
        dtype=torch.float32,
        device=device,
    )

    # NCCL initialization and kernel warm-up.
    for _ in range(warmup):
        tensor.fill_(rank + 1)
        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )
        torch.cuda.synchronize()

    dist.barrier(device_ids=[local_rank])

    local_times_ms: list[float] = []

    for _ in range(repetitions):
        # Reset outside the measured region to avoid repeated growth.
        tensor.fill_(rank + 1)
        torch.cuda.synchronize()

        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize()

        start = time.perf_counter()

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )
        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        local_times_ms.append(elapsed_ms)

    gathered_times: list[list[float] | None] = [
        None for _ in range(world_size)
    ]

    dist.all_gather_object(
        gathered_times,
        local_times_ms,
    )

    if rank != 0:
        return {}

    # A distributed operation finishes when the slowest rank finishes.
    iteration_times = [
        max(rank_times[i] for rank_times in gathered_times)
        for i in range(repetitions)
    ]

    mean_ms = statistics.mean(iteration_times)
    std_ms = (
        statistics.stdev(iteration_times)
        if len(iteration_times) > 1
        else 0.0
    )

    sorted_times = sorted(iteration_times)
    p50_ms = sorted_times[len(sorted_times) // 2]
    p95_index = min(
        len(sorted_times) - 1,
        int(len(sorted_times) * 0.95),
    )
    p95_ms = sorted_times[p95_index]

    seconds = mean_ms / 1000

    # Algorithmic bandwidth: useful tensor bytes / latency.
    algorithmic_gbps = (
        size_bytes / seconds / 1e9
    )

    # Ring all-reduce sends 2(N-1)/N times the tensor size.
    bus_gbps = (
        2
        * (world_size - 1)
        / world_size
        * size_bytes
        / seconds
        / 1e9
    )

    return {
        "world_size": world_size,
        "size_mib": size_mib,
        "size_bytes": size_bytes,
        "warmup": warmup,
        "repetitions": repetitions,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "algorithmic_gbps": algorithmic_gbps,
        "bus_gbps": bus_gbps,
    }


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = setup()

    rows: list[dict[str, float | int]] = []

    try:
        for size_mib in args.sizes_mib:
            if rank == 0:
                print(
                    f"Benchmarking world_size={world_size}, "
                    f"size={size_mib} MiB"
                )

            row = benchmark_size(
                size_mib=size_mib,
                local_rank=local_rank,
                rank=rank,
                world_size=world_size,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )

            if rank == 0:
                rows.append(row)

                print(
                    f"  {row['mean_ms']:.3f} "
                    f"± {row['std_ms']:.3f} ms, "
                    f"bus bandwidth="
                    f"{row['bus_gbps']:.2f} GB/s"
                )

        if rank == 0:
            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with args.output.open(
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

            print(f"Saved to: {args.output.resolve()}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
