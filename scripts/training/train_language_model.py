from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from llm_training_scaling.model.checkpoint import load_checkpoint, save_checkpoint
from llm_training_scaling.model.data import get_batch
from llm_training_scaling.model.transformer import TransformerLM
from llm_training_scaling.model.nn_utils import cross_entropy
from llm_training_scaling.model.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer language model on TinyStories."
    )

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------

    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path(
            "data/tinystories_tokens/"
            "tinystories_train_tokens_uint16.bin"
        ),
    )

    parser.add_argument(
        "--valid-data",
        type=Path,
        default=Path(
            "data/tinystories_tokens/"
            "tinystories_valid_tokens_uint16.bin"
        ),
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
    )

    # ----------------------------------------------------------
    # Model
    #
    # 约 15M parameters，适合 RTX 4070 Laptop。
    # ----------------------------------------------------------

    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--d-ff",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10_000.0,
    )

    # ----------------------------------------------------------
    # Optimization
    # ----------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Micro-batch size.",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-iters",
        type=int,
        default=5_000,
        help="Number of optimizer steps.",
    )

    parser.add_argument(
        "--max-lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--min-lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="Defaults to max-iters.",
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    # ----------------------------------------------------------
    # Evaluation and checkpointing
    # ----------------------------------------------------------

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--eval-iters",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/tinystories"),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume from.",
    )

    # ----------------------------------------------------------
    # Runtime
    # ----------------------------------------------------------

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = [
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "gradient_accumulation_steps",
        "max_iters",
        "eval_interval",
        "eval_iters",
        "log_interval",
        "checkpoint_interval",
    ]

    for name in positive_integer_names:
        value = getattr(args, name)

        if value <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be positive, "
                f"got {value}"
            )

    if args.d_model % args.num_heads != 0:
        raise ValueError(
            "d_model must be divisible by num_heads: "
            f"d_model={args.d_model}, "
            f"num_heads={args.num_heads}"
        )

    head_dim = args.d_model // args.num_heads

    if head_dim % 2 != 0:
        raise ValueError(
            "The attention head dimension must be even for RoPE: "
            f"head_dim={head_dim}"
        )

    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters cannot be negative.")

    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm cannot be negative.")

    if not args.train_data.exists():
        raise FileNotFoundError(
            f"Training token file not found: {args.train_data}"
        )

    if not args.valid_data.exists():
        raise FileNotFoundError(
            f"Validation token file not found: {args.valid_data}"
        )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_argument)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )

    return device


def resolve_amp_dtype(
    precision: str,
    device: torch.device,
) -> torch.dtype | None:
    """
    返回 autocast 使用的 dtype。

    None 表示不使用 autocast。
    """

    if device.type != "cuda":
        if precision not in {"auto", "float32"}:
            raise ValueError(
                f"{precision} precision currently requires CUDA."
            )

        return None

    if precision == "float32":
        return None

    if precision == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "This CUDA device/PyTorch build does not support bfloat16."
            )

        return torch.bfloat16

    if precision == "float16":
        return torch.float16

    # auto:
    # RTX 4070 支持 bfloat16 时优先使用 bfloat16。
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def make_autocast_context(
    device: torch.device,
    amp_dtype: torch.dtype | None,
):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        )

    return nullcontext()


def create_grad_scaler(enabled: bool):
    """
    bfloat16 通常不需要 GradScaler。
    float16 需要 GradScaler 防止梯度下溢。
    """

    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def namespace_to_jsonable(
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value

    return result


@torch.inference_mode()
def estimate_validation_loss(
    model: nn.Module,
    validation_data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()

    losses: list[float] = []

    for _ in range(eval_iters):
        inputs, targets = get_batch(
            dataset=validation_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        with make_autocast_context(device, amp_dtype):
            logits = model(inputs)

            # Cross-entropy 使用 float32 logits，数值更稳定。
            loss = cross_entropy(
                logits.float(),
                targets,
            )

        losses.append(float(loss.item()))

    model.train()

    return sum(losses) / len(losses)


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_iters

    validate_args(args)
    set_random_seed(args.seed)

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(
        args.precision,
        device,
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Load pre-tokenized data through memory mapping.
    # ----------------------------------------------------------

    train_data = np.memmap(
        args.train_data,
        dtype=np.uint16,
        mode="r",
    )

    validation_data = np.memmap(
        args.valid_data,
        dtype=np.uint16,
        mode="r",
    )

    if len(train_data) <= args.context_length:
        raise ValueError(
            "Training dataset is shorter than context_length."
        )

    if len(validation_data) <= args.context_length:
        raise ValueError(
            "Validation dataset is shorter than context_length."
        )

    # ----------------------------------------------------------
    # Create model and optimizer.
    #
    # 参数本身保持 float32。
    # autocast 只在 forward 中使用较低精度矩阵计算。
    # ----------------------------------------------------------

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    parameter_count = count_parameters(model)

    # 只有 float16 autocast 使用 GradScaler。
    use_grad_scaler = (
        device.type == "cuda"
        and amp_dtype == torch.float16
    )

    scaler = create_grad_scaler(
        enabled=use_grad_scaler,
    )

    start_iteration = 0

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {args.resume}"
            )

        start_iteration = load_checkpoint(
            src=args.resume,
            model=model,
            optimizer=optimizer,
        )

        print(
            f"Resumed checkpoint {args.resume} "
            f"at iteration {start_iteration:,}."
        )

    config_path = args.output_dir / "config.json"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                **namespace_to_jsonable(args),
                "resolved_device": str(device),
                "resolved_amp_dtype": (
                    str(amp_dtype)
                    if amp_dtype is not None
                    else "float32"
                ),
                "parameter_count": parameter_count,
                "train_token_count": len(train_data),
                "validation_token_count": len(validation_data),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    effective_batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
    )

    tokens_per_optimizer_step = (
        effective_batch_size
        * args.context_length
    )

    print("=" * 72)
    print("TinyStories Transformer training")
    print("=" * 72)
    print(f"Device:                {device}")
    print(
        "Autocast dtype:        "
        f"{amp_dtype if amp_dtype is not None else 'float32'}"
    )
    print(f"Parameters:            {parameter_count:,}")
    print(f"Train tokens:          {len(train_data):,}")
    print(f"Validation tokens:     {len(validation_data):,}")
    print(f"Context length:        {args.context_length}")
    print(f"Micro-batch size:      {args.batch_size}")
    print(
        "Gradient accumulation: "
        f"{args.gradient_accumulation_steps}"
    )
    print(f"Effective batch size:  {effective_batch_size}")
    print(
        "Tokens/optimizer step: "
        f"{tokens_per_optimizer_step:,}"
    )
    print(f"Start iteration:       {start_iteration:,}")
    print(f"Maximum iterations:    {args.max_iters:,}")
    print("=" * 72)

    model.train()

    log_start_time = time.perf_counter()
    tokens_since_log = 0

    last_completed_iteration = start_iteration

    try:
        for iteration in range(
            start_iteration,
            args.max_iters,
        ):
            learning_rate = get_lr_cosine_schedule(
                it=iteration,
                max_learning_rate=args.max_lr,
                min_learning_rate=args.min_lr,
                warmup_iters=args.warmup_iters,
                cosine_cycle_iters=args.cosine_cycle_iters,
            )

            set_optimizer_learning_rate(
                optimizer,
                learning_rate,
            )

            optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0

            # --------------------------------------------------
            # Gradient accumulation
            # --------------------------------------------------

            for _ in range(
                args.gradient_accumulation_steps
            ):
                inputs, targets = get_batch(
                    dataset=train_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device,
                )

                with make_autocast_context(
                    device,
                    amp_dtype,
                ):
                    logits = model(inputs)

                    raw_loss = cross_entropy(
                        logits.float(),
                        targets,
                    )

                    loss_for_backward = (
                        raw_loss
                        / args.gradient_accumulation_steps
                    )

                accumulated_loss += float(
                    raw_loss.detach().item()
                )

                if use_grad_scaler:
                    scaler.scale(
                        loss_for_backward
                    ).backward()
                else:
                    loss_for_backward.backward()

            # float16 时先 unscale，之后才能正确裁剪梯度。
            if use_grad_scaler:
                scaler.unscale_(optimizer)

            gradient_clipping(
                model.parameters(),
                max_l2_norm=args.max_grad_norm,
            )

            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            completed_iteration = iteration + 1
            last_completed_iteration = completed_iteration
            tokens_since_log += tokens_per_optimizer_step

            mean_train_loss = (
                accumulated_loss
                / args.gradient_accumulation_steps
            )

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            should_log = (
                completed_iteration == 1
                or completed_iteration % args.log_interval == 0
            )

            if should_log:
                synchronize_cuda(device)

                now = time.perf_counter()
                elapsed = now - log_start_time

                tokens_per_second = (
                    tokens_since_log / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"iter {completed_iteration:6d}/"
                    f"{args.max_iters} | "
                    f"train loss {mean_train_loss:.4f} | "
                    f"lr {learning_rate:.3e} | "
                    f"{tokens_per_second:,.0f} tok/s"
                )

                log_start_time = now
                tokens_since_log = 0

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            should_evaluate = (
                completed_iteration % args.eval_interval == 0
                or completed_iteration == args.max_iters
            )

            if should_evaluate:
                validation_loss = estimate_validation_loss(
                    model=model,
                    validation_data=validation_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    eval_iters=args.eval_iters,
                    device=device,
                    amp_dtype=amp_dtype,
                )

                print(
                    f"iter {completed_iteration:6d} | "
                    f"validation loss {validation_loss:.4f}"
                )

                # Validation 本身需要时间，因此重新开始吞吐计时。
                synchronize_cuda(device)
                log_start_time = time.perf_counter()
                tokens_since_log = 0

            # --------------------------------------------------
            # Periodic checkpoint
            # --------------------------------------------------

            should_checkpoint = (
                completed_iteration
                % args.checkpoint_interval
                == 0
            )

            if should_checkpoint:
                checkpoint_path = (
                    checkpoint_dir
                    / f"step_{completed_iteration:06d}.pt"
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_iteration,
                    out=checkpoint_path,
                )

                latest_path = (
                    checkpoint_dir
                    / "latest.pt"
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=completed_iteration,
                    out=latest_path,
                )

                print(
                    f"Saved checkpoint: {checkpoint_path}"
                )

    except KeyboardInterrupt:
        interrupted_path = (
            checkpoint_dir
            / f"interrupted_step_"
            f"{last_completed_iteration:06d}.pt"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=last_completed_iteration,
            out=interrupted_path,
        )

        print()
        print(
            "Training interrupted. "
            f"Checkpoint saved to {interrupted_path}"
        )

        return

    final_checkpoint_path = (
        checkpoint_dir
        / "final.pt"
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=last_completed_iteration,
        out=final_checkpoint_path,
    )

    print("=" * 72)
    print("Training completed.")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
