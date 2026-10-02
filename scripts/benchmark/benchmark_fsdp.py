from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llm_training_scaling.systems.benchmark_model import BasicsTransformerLM
from llm_training_scaling.systems.fsdp_prefetch import PrefetchFullyShardedDataParallel as FullyShardedDataParallel


XL_CONFIG = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute-dtype",
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
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


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def model_parameter_bytes(
    model: torch.nn.Module,
) -> int:
    return sum(tensor_bytes(p) for p in model.parameters())


def model_gradient_bytes(
    model: torch.nn.Module,
) -> int:
    return sum(
        tensor_bytes(p.grad)
        for p in model.parameters()
        if p.grad is not None
    )


def recursive_tensor_bytes(
    value: Any,
    seen: set[int] | None = None,
) -> int:
    if seen is None:
        seen = set()

    if isinstance(value, torch.Tensor):
        value_id = id(value)
        if value_id in seen:
            return 0
        seen.add(value_id)
        return tensor_bytes(value)

    if isinstance(value, dict):
        return sum(
            recursive_tensor_bytes(v, seen)
            for v in value.values()
        )

    if isinstance(value, (list, tuple)):
        return sum(
            recursive_tensor_bytes(v, seen)
            for v in value
        )

    return 0


def optimizer_state_bytes(
    optimizer: torch.optim.Optimizer,
) -> int:
    return recursive_tensor_bytes(optimizer.state)


def distributed_max(
    value: int | float,
    device: torch.device,
) -> float:
    tensor = torch.tensor(
        float(value),
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def memory_snapshot(
    checkpoint: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.synchronize(device)

    return {
        "checkpoint": checkpoint,
        "allocated_gib": distributed_max(
            torch.cuda.memory_allocated(device),
            device,
        ) / 1024**3,
        "reserved_gib": distributed_max(
            torch.cuda.memory_reserved(device),
            device,
        ) / 1024**3,
        "phase_peak_gib": distributed_max(
            torch.cuda.max_memory_allocated(device),
            device,
        ) / 1024**3,
        "local_parameter_gib": distributed_max(
            model_parameter_bytes(model),
            device,
        ) / 1024**3,
        "local_gradient_gib": distributed_max(
            model_gradient_bytes(model),
            device,
        ) / 1024**3,
        "optimizer_state_gib": distributed_max(
            optimizer_state_bytes(optimizer),
            device,
        ) / 1024**3,
    }


def gather_slowest_rank_times(
    local_times: list[float],
    world_size: int,
) -> list[float]:
    gathered: list[list[float] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered, local_times)

    return [
        max(
            rank_times[index]
            for rank_times in gathered
            if rank_times is not None
        )
        for index in range(len(local_times))
    ]


def run_step(
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    phase: str,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.nvtx.range(f"{phase}_step"):
        with torch.cuda.nvtx.range("forward_and_loss"):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size).float(),
                targets.reshape(-1),
            )

        with torch.cuda.nvtx.range("backward"):
            loss.backward()

        with torch.cuda.nvtx.range(
            "finish_gradient_synchronization"
        ):
            model.finish_gradient_synchronization()

        with torch.cuda.nvtx.range("optimizer_step"):
            optimizer.step()

    return loss


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    try:
        if world_size != 2:
            raise ValueError("This benchmark expects 2 GPUs")

        if args.global_batch_size % world_size != 0:
            raise ValueError(
                "global batch size must divide world size"
            )

        local_batch_size = (
            args.global_batch_size // world_size
        )

        compute_dtype = (
            None
            if args.compute_dtype == "fp32"
            else torch.bfloat16
        )

        torch.manual_seed(1234 + rank)
        torch.cuda.manual_seed_all(1234 + rank)
        torch.set_float32_matmul_precision("high")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        if rank == 0:
            print("Building XL model and applying FSDP...")

        base_model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            **XL_CONFIG,
        ).to(device)

        model = FullyShardedDataParallel(
            base_model,
            compute_dtype=compute_dtype,
        )

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.1,
            foreach=False,
        )

        dist.barrier(device_ids=[local_rank])

        rows: list[dict[str, Any]] = []

        rows.append(
            memory_snapshot(
                "after_fsdp_initialization",
                model,
                optimizer,
                device,
            )
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

        # First step: explicitly separate memory checkpoints.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)

        with torch.cuda.nvtx.range("memory_forward_and_loss"):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, args.vocab_size).float(),
                targets.reshape(-1),
            )

        with torch.cuda.nvtx.range("memory_backward"):
            loss.backward()

        model.finish_gradient_synchronization()
        torch.cuda.synchronize(device)

        rows.append(
            memory_snapshot(
                "before_first_optimizer_step",
                model,
                optimizer,
                device,
            )
        )

        torch.cuda.reset_peak_memory_stats(device)

        with torch.cuda.nvtx.range(
            "memory_first_optimizer_step"
        ):
            optimizer.step()

        torch.cuda.synchronize(device)

        rows.append(
            memory_snapshot(
                "after_first_optimizer_step",
                model,
                optimizer,
                device,
            )
        )

        if rank == 0:
            print("\n=== FSDP memory checkpoints ===")
            for row in rows:
                print(
                    f"{row['checkpoint']}: "
                    f"allocated={row['allocated_gib']:.2f} GiB, "
                    f"peak={row['phase_peak_gib']:.2f} GiB, "
                    f"params={row['local_parameter_gib']:.2f} GiB, "
                    f"grads={row['local_gradient_gib']:.2f} GiB, "
                    f"optimizer_state="
                    f"{row['optimizer_state_gib']:.2f} GiB"
                )

        local_times_ms: list[float] = []
        total_steps = args.warmup + args.repetitions

        for step in range(total_steps):
            phase = (
                "warmup"
                if step < args.warmup
                else "measurement"
            )

            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize(device)

            start = time.perf_counter()

            loss = run_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                vocab_size=args.vocab_size,
                phase=phase,
            )

            torch.cuda.synchronize(device)
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            if step >= args.warmup:
                local_times_ms.append(elapsed_ms)

            if rank == 0:
                print(
                    f"{phase} {step + 1}/{total_steps}: "
                    f"{elapsed_ms:.2f} ms, "
                    f"loss={float(loss.detach()):.4f}"
                )

        slowest_times = gather_slowest_rank_times(
            local_times_ms,
            world_size,
        )

        step_mean_ms = statistics.mean(slowest_times)
        step_std_ms = (
            statistics.stdev(slowest_times)
            if len(slowest_times) > 1
            else 0.0
        )

        for row in rows:
            row.update(
                {
                    "implementation": "forward_prefetch_two_layers",
                    "world_size": world_size,
                    "compute_dtype": args.compute_dtype,
                    "global_batch_size": args.global_batch_size,
                    "local_batch_size": local_batch_size,
                    "context_length": args.context_length,
                    "step_mean_ms": step_mean_ms,
                    "step_std_ms": step_std_ms,
                }
            )

        if rank == 0:
            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            fieldnames = [
                "implementation",
                "checkpoint",
                "world_size",
                "compute_dtype",
                "global_batch_size",
                "local_batch_size",
                "context_length",
                "allocated_gib",
                "reserved_gib",
                "phase_peak_gib",
                "local_parameter_gib",
                "local_gradient_gib",
                "optimizer_state_gib",
                "step_mean_ms",
                "step_std_ms",
            ]

            with args.output.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
                writer.writerows(rows)

            print("\n=== FSDP steady-state timing ===")
            print(
                f"Step time: "
                f"{step_mean_ms:.2f} ± {step_std_ms:.2f} ms"
            )
            print(f"Saved to: {args.output.resolve()}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
