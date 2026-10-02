from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llm_training_scaling.systems.benchmark_model import BasicsTransformerLM
from llm_training_scaling.systems.ddp import FlatGradientDDP, NaiveDDP, OverlapDDP


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=MODEL_CONFIGS,
        default="xl",
    )
    parser.add_argument(
        "--implementation",
        choices=["naive", "flat", "overlap"],
        default="naive",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
    )
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


def gather_slowest_rank_times(
    local_times: list[float],
    world_size: int,
) -> list[float]:
    gathered: list[list[float] | None] = [
        None for _ in range(world_size)
    ]

    dist.all_gather_object(gathered, local_times)

    assert all(item is not None for item in gathered)

    return [
        max(rank_times[index] for rank_times in gathered)
        for index in range(len(local_times))
    ]


def mean_std(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values)
        if len(values) > 1
        else 0.0,
    )


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = setup()

    try:
        if args.global_batch_size % world_size != 0:
            raise ValueError(
                "global batch size must be divisible by world size"
            )

        device = torch.device(f"cuda:{local_rank}")
        local_batch_size = (
            args.global_batch_size // world_size
        )

        torch.manual_seed(1234 + rank)
        torch.cuda.manual_seed_all(1234 + rank)
        torch.set_float32_matmul_precision("high")

        config = MODEL_CONFIGS[args.model]

        if rank == 0:
            print(
                f"Building {args.model} model on "
                f"{world_size} GPUs..."
            )

        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
        ).to(device)

        ddp_classes = {
            "naive": NaiveDDP,
            "flat": FlatGradientDDP,
            "overlap": OverlapDDP,
        }
        ddp_class = ddp_classes[args.implementation]
        ddp_model = ddp_class(model)

        # SGD avoids Adam optimizer-state memory from obscuring the
        # DDP communication experiment.
        optimizer = torch.optim.SGD(
            ddp_model.parameters(),
            lr=1e-3,
        )

        inputs = torch.randint(
            0,
            args.vocab_size,
            (
                local_batch_size,
                args.context_length,
            ),
            device=device,
        )

        targets = torch.randint(
            0,
            args.vocab_size,
            (
                local_batch_size,
                args.context_length,
            ),
            device=device,
        )

        if args.precision == "bf16":
            autocast_context = lambda: torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        else:
            autocast_context = nullcontext

        parameter_count = sum(
            parameter.numel()
            for parameter in ddp_model.parameters()
        )

        gradient_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in ddp_model.parameters()
            if parameter.requires_grad
        )

        if rank == 0:
            print(
                f"Parameters: {parameter_count / 1e9:.3f}B"
            )
            print(
                f"Gradient volume per rank: "
                f"{gradient_bytes / 1024**3:.2f} GiB"
            )
            print(
                f"Global batch={args.global_batch_size}, "
                f"local batch={local_batch_size}, "
                f"context={args.context_length}, "
                f"precision={args.precision}"
            )

        total_times_ms: list[float] = []
        communication_times_ms: list[float] = []
        losses: list[float] = []

        total_steps = args.warmup + args.repetitions

        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)

            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize()

            total_start = time.perf_counter()

            phase = (
                "warmup"
                if step < args.warmup
                else "measurement"
            )

            with torch.cuda.nvtx.range(f"{phase}_step"):
                with torch.cuda.nvtx.range("forward_and_loss"):
                    with autocast_context():
                        logits = ddp_model(inputs)

                        loss = F.cross_entropy(
                            logits.reshape(-1, args.vocab_size),
                            targets.reshape(-1),
                        )

                with torch.cuda.nvtx.range("backward"):
                    loss.backward()

                # Finish backward before isolating remaining
                # communication tail time.
                torch.cuda.synchronize()

                communication_start = time.perf_counter()

                with torch.cuda.nvtx.range(
                    "gradient_synchronization"
                ):
                    ddp_model.finish_gradient_synchronization()

                torch.cuda.synchronize()

                communication_ms = (
                    time.perf_counter() - communication_start
                ) * 1000

                with torch.cuda.nvtx.range("optimizer_step"):
                    optimizer.step()

                torch.cuda.synchronize()

            total_ms = (
                time.perf_counter() - total_start
            ) * 1000

            if step >= args.warmup:
                total_times_ms.append(total_ms)
                communication_times_ms.append(
                    communication_ms
                )
                losses.append(float(loss.detach()))

            if rank == 0:
                phase = (
                    "warmup"
                    if step < args.warmup
                    else "measure"
                )
                print(
                    f"{phase} {step + 1}/{total_steps}: "
                    f"total={total_ms:.2f} ms, "
                    f"communication={communication_ms:.2f} ms, "
                    f"loss={float(loss.detach()):.4f}"
                )

        slowest_total = gather_slowest_rank_times(
            total_times_ms,
            world_size,
        )
        slowest_communication = gather_slowest_rank_times(
            communication_times_ms,
            world_size,
        )

        if rank == 0:
            total_mean, total_std = mean_std(
                slowest_total
            )
            comm_mean, comm_std = mean_std(
                slowest_communication
            )

            communication_fraction = (
                comm_mean / total_mean
            )

            row = {
                "implementation": {
                    "naive": "naive_per_parameter",
                    "flat": "flat_single_all_reduce",
                    "overlap": "overlap_individual_parameters",
                }[args.implementation],
                "model": args.model,
                "world_size": world_size,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "context_length": args.context_length,
                "vocab_size": args.vocab_size,
                "precision": args.precision,
                "parameter_count": parameter_count,
                "gradient_gib": gradient_bytes / 1024**3,
                "warmup": args.warmup,
                "repetitions": args.repetitions,
                "total_mean_ms": total_mean,
                "total_std_ms": total_std,
                "communication_mean_ms": comm_mean,
                "communication_std_ms": comm_std,
                "communication_fraction": communication_fraction,
            }

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
                    fieldnames=list(row.keys()),
                )
                writer.writeheader()
                writer.writerow(row)

            print("\n=== Naive DDP result ===")
            print(
                f"Training step: "
                f"{total_mean:.2f} ± {total_std:.2f} ms"
            )
            print(
                f"Gradient communication: "
                f"{comm_mean:.2f} ± {comm_std:.2f} ms"
            )
            print(
                f"Communication fraction: "
                f"{communication_fraction * 100:.1f}%"
            )
            print(f"Saved to: {args.output.resolve()}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
