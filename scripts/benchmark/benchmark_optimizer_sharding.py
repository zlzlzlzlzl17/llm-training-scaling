from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from llm_training_scaling.systems.benchmark_model import BasicsTransformerLM
from llm_training_scaling.systems.ddp import OverlapDDP
from llm_training_scaling.systems.sharded_optimizer import ShardedOptimizer


XL_CONFIG = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        choices=["regular", "sharded"],
        required=True,
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
        "--warmup",
        type=int,
        default=2,
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


def setup_distributed() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    return local_rank, rank, world_size


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(
        tensor_nbytes(parameter)
        for parameter in model.parameters()
    )


def gradient_bytes(model: torch.nn.Module) -> int:
    return sum(
        tensor_nbytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def object_tensor_bytes(
    value: Any,
    seen: set[int] | None = None,
) -> int:
    if seen is None:
        seen = set()

    if isinstance(value, torch.Tensor):
        object_id = id(value)

        if object_id in seen:
            return 0

        seen.add(object_id)
        return tensor_nbytes(value)

    if isinstance(value, dict):
        return sum(
            object_tensor_bytes(item, seen)
            for item in value.values()
        )

    if isinstance(value, (list, tuple)):
        return sum(
            object_tensor_bytes(item, seen)
            for item in value
        )

    return 0


def optimizer_state_bytes(
    optimizer: torch.optim.Optimizer,
) -> int:
    return object_tensor_bytes(optimizer.state)


def distributed_max(
    value: int | float,
    device: torch.device,
) -> float:
    tensor = torch.tensor(
        float(value),
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.MAX,
    )

    return float(tensor.item())


def memory_snapshot(
    label: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float | str]:
    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    phase_peak = torch.cuda.max_memory_allocated(device)

    params = parameter_bytes(model)
    gradients = gradient_bytes(model)
    optimizer_state = optimizer_state_bytes(optimizer)

    return {
        "checkpoint": label,
        "allocated_gib": distributed_max(
            allocated,
            device,
        ) / 1024**3,
        "reserved_gib": distributed_max(
            reserved,
            device,
        ) / 1024**3,
        "phase_peak_gib": distributed_max(
            phase_peak,
            device,
        ) / 1024**3,
        "parameter_gib": distributed_max(
            params,
            device,
        ) / 1024**3,
        "gradient_gib": distributed_max(
            gradients,
            device,
        ) / 1024**3,
        "optimizer_state_gib": distributed_max(
            optimizer_state,
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

    dist.all_gather_object(
        gathered,
        local_times,
    )

    return [
        max(
            rank_times[index]
            for rank_times in gathered
            if rank_times is not None
        )
        for index in range(len(local_times))
    ]


def train_step(
    model: OverlapDDP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()

    return loss


def main() -> None:
    args = parse_args()

    local_rank, rank, world_size = (
        setup_distributed()
    )

    device = torch.device(f"cuda:{local_rank}")

    try:
        if world_size != 2:
            raise ValueError(
                "This benchmark expects exactly 2 GPUs"
            )

        if args.global_batch_size % world_size != 0:
            raise ValueError(
                "global batch size must be divisible "
                "by world size"
            )

        local_batch_size = (
            args.global_batch_size // world_size
        )

        torch.manual_seed(1234 + rank)
        torch.cuda.manual_seed_all(1234 + rank)
        torch.set_float32_matmul_precision("high")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        if rank == 0:
            print("Building XL model...")

        base_model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            **XL_CONFIG,
        ).to(device)

        model = OverlapDDP(base_model)

        optimizer_kwargs = {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.1,
            # Avoid foreach temporary tensor-list memory
            # obscuring the optimizer-state comparison.
            "foreach": False,
        }

        if args.optimizer == "regular":
            optimizer: torch.optim.Optimizer = (
                torch.optim.AdamW(
                    model.parameters(),
                    **optimizer_kwargs,
                )
            )
        else:
            optimizer = ShardedOptimizer(
                model.parameters(),
                torch.optim.AdamW,
                **optimizer_kwargs,
            )

        dist.barrier(device_ids=[local_rank])

        rows: list[dict[str, Any]] = []

        rows.append(
            memory_snapshot(
                label="after_model_initialization",
                model=model,
                optimizer=optimizer,
                device=device,
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

        # First step is used for memory accounting. AdamW state
        # does not exist until optimizer.step() is called.
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.reset_peak_memory_stats(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(inputs)

            loss = F.cross_entropy(
                logits.reshape(-1, args.vocab_size),
                targets.reshape(-1),
            )

        loss.backward()
        model.finish_gradient_synchronization()
        torch.cuda.synchronize(device)

        rows.append(
            memory_snapshot(
                label="before_first_optimizer_step",
                model=model,
                optimizer=optimizer,
                device=device,
            )
        )

        # Measure optimizer-step peak separately.
        torch.cuda.reset_peak_memory_stats(device)

        optimizer.step()
        torch.cuda.synchronize(device)

        rows.append(
            memory_snapshot(
                label="after_first_optimizer_step",
                model=model,
                optimizer=optimizer,
                device=device,
            )
        )

        if rank == 0:
            print(
                f"Parameters: "
                f"{rows[0]['parameter_gib']:.2f} GiB"
            )
            print(
                f"Optimizer implementation: "
                f"{args.optimizer}"
            )

            print("\n=== Memory checkpoints ===")

            for row in rows:
                print(
                    f"{row['checkpoint']}: "
                    f"allocated={row['allocated_gib']:.2f} GiB, "
                    f"phase_peak={row['phase_peak_gib']:.2f} GiB, "
                    f"gradients={row['gradient_gib']:.2f} GiB, "
                    f"optimizer_state="
                    f"{row['optimizer_state_gib']:.2f} GiB"
                )

        # State has now been initialized. Run steady-state timing.
        local_step_times_ms: list[float] = []
        total_steps = args.warmup + args.repetitions

        for step in range(total_steps):
            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize(device)

            start = time.perf_counter()

            loss = train_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                vocab_size=args.vocab_size,
            )

            torch.cuda.synchronize(device)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            if step >= args.warmup:
                local_step_times_ms.append(elapsed_ms)

            if rank == 0:
                phase = (
                    "warmup"
                    if step < args.warmup
                    else "measure"
                )

                print(
                    f"{phase} {step + 1}/{total_steps}: "
                    f"{elapsed_ms:.2f} ms, "
                    f"loss={float(loss.detach()):.4f}"
                )

        slowest_times = gather_slowest_rank_times(
            local_step_times_ms,
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
                    "optimizer": args.optimizer,
                    "world_size": world_size,
                    "global_batch_size": (
                        args.global_batch_size
                    ),
                    "local_batch_size": local_batch_size,
                    "context_length": (
                        args.context_length
                    ),
                    "precision": "bf16_compute_fp32_master",
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
                "optimizer",
                "checkpoint",
                "world_size",
                "global_batch_size",
                "local_batch_size",
                "context_length",
                "precision",
                "allocated_gib",
                "reserved_gib",
                "phase_peak_gib",
                "parameter_gib",
                "gradient_gib",
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

            print("\n=== Steady-state timing ===")
            print(
                f"Step time: "
                f"{step_mean_ms:.2f} ± "
                f"{step_std_ms:.2f} ms"
            )
            print(
                f"Saved to: {args.output.resolve()}"
            )

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
