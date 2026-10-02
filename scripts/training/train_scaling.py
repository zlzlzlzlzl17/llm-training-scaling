from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import llm_training_scaling.model.transformer as model_impl


def fast_scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use PyTorch SDPA; the model only supplies a causal lower-triangular mask."""
    return F.scaled_dot_product_attention(
        queries,
        keys,
        values,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=mask is not None,
    )


class SequentialTokenStream:
    """Read a fixed token stream once, from left to right, without epoching."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        batch_size: int,
        context_length: int,
        device: torch.device,
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.tokens_per_batch = batch_size * context_length
        self.offset = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.offset
        stop = start + self.tokens_per_batch

        if stop >= len(self.data):
            raise RuntimeError(
                "Training token stream was exhausted: "
                f"need index {stop}, but file has {len(self.data)} tokens."
            )

        window = np.asarray(
            self.data[start : stop + 1],
            dtype=np.int64,
        )

        inputs_np = window[:-1].reshape(
            self.batch_size,
            self.context_length,
        )
        targets_np = window[1:].reshape(
            self.batch_size,
            self.context_length,
        )

        self.offset = stop

        inputs = torch.from_numpy(inputs_np).to(
            device=self.device,
            dtype=torch.long,
        )
        targets = torch.from_numpy(targets_np).to(
            device=self.device,
            dtype=torch.long,
        )
        return inputs, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fixed-order DCLM runs for scaling-law experiments."
    )

    parser.add_argument("--run-name", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)

    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--total-train-tokens", type=int, required=True)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--n-evals", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--final-lr-frac", type=float, default=0.1)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "manual"],
        default="sdpa",
    )
    parser.add_argument(
        "--precision",
        choices=["bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-final-checkpoint", action="store_true")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = [
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "total_train_tokens",
        "validation_batch_size",
        "n_evals",
        "log_interval",
    ]
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if not 0.0 <= args.warmup_frac < 1.0:
        raise ValueError("warmup_frac must be in [0, 1)")
    if not 0.0 <= args.final_lr_frac <= 1.0:
        raise ValueError("final_lr_frac must be in [0, 1]")
    if args.peak_lr <= 0:
        raise ValueError("peak_lr must be positive")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if not args.train_data.exists():
        raise FileNotFoundError(args.train_data)
    if not args.valid_data.exists():
        raise FileNotFoundError(args.valid_data)

    tokens_per_step = (
        args.micro_batch_size
        * args.gradient_accumulation_steps
        * args.context_length
    )
    if args.total_train_tokens % tokens_per_step != 0:
        raise ValueError(
            "total_train_tokens must be divisible by tokens_per_optimizer_step: "
            f"{args.total_train_tokens} % {tokens_per_step} != 0"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(precision: str):
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int, int, int]:
    total = 0
    embedding = 0
    lm_head = 0

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if name == "token_embeddings.weight":
            embedding += count
        elif name == "lm_head.weight":
            lm_head += count

    non_embedding = total - embedding - lm_head
    return total, non_embedding, embedding, lm_head


def make_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, bool]:
    kwargs: dict[str, Any] = {
        "lr": args.peak_lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.adam_eps,
        "weight_decay": args.weight_decay,
    }

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            fused=True,
            **kwargs,
        )
        return optimizer, True
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **kwargs,
        )
        return optimizer, False


def learning_rate_at_step(
    step_index: int,
    *,
    max_steps: int,
    peak_lr: float,
    final_lr_frac: float,
    warmup_frac: float,
) -> float:
    warmup_steps = int(round(max_steps * warmup_frac))
    warmup_steps = min(max(warmup_steps, 0), max_steps - 1)

    if warmup_steps > 0 and step_index < warmup_steps:
        return peak_lr * (step_index + 1) / warmup_steps

    final_lr = peak_lr * final_lr_frac
    decay_steps = max_steps - warmup_steps
    decay_index = step_index - warmup_steps

    if decay_steps <= 1:
        progress = 1.0
    else:
        progress = min(max(decay_index / (decay_steps - 1), 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (peak_lr - final_lr) * cosine


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def evaluation_steps(max_steps: int, n_evals: int) -> set[int]:
    steps = {
        max(1, math.ceil(index * max_steps / n_evals))
        for index in range(1, n_evals + 1)
    }
    steps.add(max_steps)
    return steps


def numpy_window_to_tensors(
    data: np.ndarray,
    *,
    start: int,
    token_count: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = np.asarray(
        data[start : start + token_count + 1],
        dtype=np.int64,
    )
    inputs_np = window[:-1].reshape(batch_size, sequence_length)
    targets_np = window[1:].reshape(batch_size, sequence_length)
    return (
        torch.from_numpy(inputs_np).to(device=device, dtype=torch.long),
        torch.from_numpy(targets_np).to(device=device, dtype=torch.long),
    )


@torch.inference_mode()
def evaluate_validation_loss(
    model: torch.nn.Module,
    validation_data: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
    precision: str,
) -> tuple[float, int]:
    model.eval()

    total_targets = len(validation_data) - 1
    full_sequences = total_targets // context_length
    remainder = total_targets % context_length

    loss_sum = 0.0
    scored_tokens = 0
    sequence_index = 0

    while sequence_index < full_sequences:
        current_batch = min(batch_size, full_sequences - sequence_index)
        token_count = current_batch * context_length
        start = sequence_index * context_length
        inputs, targets = numpy_window_to_tensors(
            validation_data,
            start=start,
            token_count=token_count,
            batch_size=current_batch,
            sequence_length=context_length,
            device=device,
        )
        with autocast_context(precision):
            logits = model(inputs)
            batch_loss_sum = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
        loss_sum += float(batch_loss_sum.item())
        scored_tokens += token_count
        sequence_index += current_batch
        del inputs, targets, logits, batch_loss_sum

    if remainder > 0:
        start = full_sequences * context_length
        inputs, targets = numpy_window_to_tensors(
            validation_data,
            start=start,
            token_count=remainder,
            batch_size=1,
            sequence_length=remainder,
            device=device,
        )
        with autocast_context(precision):
            logits = model(inputs)
            remainder_loss_sum = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )
        loss_sum += float(remainder_loss_sum.item())
        scored_tokens += remainder
        del inputs, targets, logits, remainder_loss_sum

    model.train()
    return loss_sum / scored_tokens, scored_tokens


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()


def main() -> None:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA.")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16.")

    if args.attention_backend == "sdpa":
        model_impl.scaled_dot_product_attention = fast_scaled_dot_product_attention

    seed_everything(args.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    config_path = args.output_dir / "config.json"
    result_path = args.output_dir / "result.json"
    metrics_path.unlink(missing_ok=True)

    train_data = np.memmap(args.train_data, dtype="<u2", mode="r")
    validation_data = np.memmap(args.valid_data, dtype="<u2", mode="r")

    if len(train_data) <= args.total_train_tokens:
        raise ValueError(
            "Training file must contain at least total_train_tokens + 1 IDs: "
            f"file={len(train_data):,}, requested={args.total_train_tokens:,}"
        )
    if len(validation_data) <= 1:
        raise ValueError("Validation file is too short.")

    model = model_impl.TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eps=args.rms_norm_eps,
        device=device,
        dtype=torch.float32,
    )

    total_parameters, non_embedding_parameters, embedding_parameters, lm_head_parameters = (
        count_parameters(model)
    )
    approximate_non_embedding_parameters = (
        12 * args.num_layers * args.d_model**2
    )

    optimizer, fused_optimizer = make_optimizer(model, args)

    tokens_per_micro_batch = args.micro_batch_size * args.context_length
    tokens_per_optimizer_step = (
        tokens_per_micro_batch * args.gradient_accumulation_steps
    )
    max_steps = args.total_train_tokens // tokens_per_optimizer_step
    eval_steps = evaluation_steps(max_steps, args.n_evals)

    stream = SequentialTokenStream(
        train_data,
        batch_size=args.micro_batch_size,
        context_length=args.context_length,
        device=device,
    )

    config = {
        **jsonable_args(args),
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "fused_optimizer": fused_optimizer,
        "total_parameters": total_parameters,
        "non_embedding_parameters": non_embedding_parameters,
        "embedding_parameters": embedding_parameters,
        "lm_head_parameters": lm_head_parameters,
        "approximate_non_embedding_parameters": approximate_non_embedding_parameters,
        "tokens_per_micro_batch": tokens_per_micro_batch,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "max_steps": max_steps,
        "validation_token_ids": len(validation_data),
        "validation_prediction_tokens": len(validation_data) - 1,
        "estimated_flops_non_embedding": (
            6 * non_embedding_parameters * args.total_train_tokens
        ),
        "estimated_flops_total_parameters": (
            6 * total_parameters * args.total_train_tokens
        ),
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("DCLM scaling-law training")
    print("=" * 80)
    print(f"Run:                         {args.run_name}")
    print(f"GPU:                         {torch.cuda.get_device_name(0)}")
    print(f"Attention backend:           {args.attention_backend}")
    print(f"Precision:                   {args.precision}")
    print(f"Total parameters:            {total_parameters:,}")
    print(f"Non-embedding parameters:    {non_embedding_parameters:,}")
    print(f"Approx. non-embedding (12Ld²): {approximate_non_embedding_parameters:,}")
    print(f"Training tokens:             {args.total_train_tokens:,}")
    print(f"Tokens/optimizer step:       {tokens_per_optimizer_step:,}")
    print(f"Optimizer steps:             {max_steps:,}")
    print(f"Peak LR:                     {args.peak_lr:.3e}")
    print(f"Fused AdamW:                 {fused_optimizer}")
    print("=" * 80)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    validation_seconds = 0.0
    last_log_time = wall_start
    last_log_tokens = 0
    final_validation_loss: float | None = None
    validation_history: list[dict[str, Any]] = []

    try:
        for step_index in range(max_steps):
            learning_rate = learning_rate_at_step(
                step_index,
                max_steps=max_steps,
                peak_lr=args.peak_lr,
                final_lr_frac=args.final_lr_frac,
                warmup_frac=args.warmup_frac,
            )
            set_learning_rate(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for _ in range(args.gradient_accumulation_steps):
                inputs, targets = stream.next_batch()
                with autocast_context(args.precision):
                    logits = model(inputs)
                    raw_loss = F.cross_entropy(
                        logits.reshape(-1, args.vocab_size),
                        targets.reshape(-1),
                    )
                    backward_loss = raw_loss / args.gradient_accumulation_steps
                backward_loss.backward()
                accumulated_loss += float(raw_loss.detach().item())
                del inputs, targets, logits, raw_loss, backward_loss

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            optimizer.step()

            completed_step = step_index + 1
            completed_tokens = completed_step * tokens_per_optimizer_step
            mean_train_loss = accumulated_loss / args.gradient_accumulation_steps

            if completed_step == 1 or completed_step % args.log_interval == 0:
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                interval_tokens = completed_tokens - last_log_tokens
                interval_seconds = now - last_log_time
                throughput = interval_tokens / interval_seconds
                record = {
                    "type": "train",
                    "step": completed_step,
                    "tokens": completed_tokens,
                    "train_loss": mean_train_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clip": float(gradient_norm.item()),
                    "tokens_per_second": throughput,
                    "wall_time_seconds": now - wall_start,
                }
                append_jsonl(metrics_path, record)
                print(
                    f"step {completed_step:6d}/{max_steps} | "
                    f"tokens {completed_tokens:,} | "
                    f"loss {mean_train_loss:.4f} | "
                    f"lr {learning_rate:.3e} | "
                    f"{throughput:,.0f} tok/s",
                    flush=True,
                )
                last_log_time = now
                last_log_tokens = completed_tokens

            if completed_step in eval_steps:
                torch.cuda.synchronize(device)
                eval_start = time.perf_counter()
                validation_loss, scored_tokens = evaluate_validation_loss(
                    model,
                    validation_data,
                    context_length=args.context_length,
                    batch_size=args.validation_batch_size,
                    vocab_size=args.vocab_size,
                    device=device,
                    precision=args.precision,
                )
                torch.cuda.synchronize(device)
                eval_duration = time.perf_counter() - eval_start
                validation_seconds += eval_duration
                final_validation_loss = validation_loss
                val_record = {
                    "type": "validation",
                    "step": completed_step,
                    "tokens": completed_tokens,
                    "validation_loss": validation_loss,
                    "scored_validation_tokens": scored_tokens,
                    "evaluation_seconds": eval_duration,
                }
                append_jsonl(metrics_path, val_record)
                validation_history.append(val_record)
                print(
                    f"step {completed_step:6d} | "
                    f"validation loss {validation_loss:.4f} | "
                    f"{scored_tokens:,} tokens | "
                    f"{eval_duration:.1f}s",
                    flush=True,
                )
                last_log_time = time.perf_counter()
                last_log_tokens = completed_tokens

    except BaseException as error:
        torch.cuda.synchronize(device)
        failure = {
            "status": "failed",
            "run_name": args.run_name,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        result_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise

    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    training_seconds = max(wall_seconds - validation_seconds, 1e-9)
    peak_memory_gib = torch.cuda.max_memory_allocated(device) / 1024**3

    if final_validation_loss is None:
        raise RuntimeError("No validation result was produced.")

    result = {
        "status": "completed",
        "run_name": args.run_name,
        "total_parameters": total_parameters,
        "non_embedding_parameters": non_embedding_parameters,
        "approximate_non_embedding_parameters": approximate_non_embedding_parameters,
        "embedding_parameters": embedding_parameters,
        "lm_head_parameters": lm_head_parameters,
        "total_train_tokens": args.total_train_tokens,
        "estimated_flops_non_embedding": (
            6 * non_embedding_parameters * args.total_train_tokens
        ),
        "estimated_flops_total_parameters": (
            6 * total_parameters * args.total_train_tokens
        ),
        "final_validation_loss": final_validation_loss,
        "validation_perplexity": math.exp(min(final_validation_loss, 50.0)),
        "validation_history": validation_history,
        "optimizer_steps": max_steps,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "wall_seconds": wall_seconds,
        "validation_seconds": validation_seconds,
        "training_seconds": training_seconds,
        "training_tokens_per_second": args.total_train_tokens / training_seconds,
        "peak_allocated_gpu_memory_gib": peak_memory_gib,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "fused_optimizer": fused_optimizer,
        "architecture": {
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "head_dim": args.d_model // args.num_heads,
            "d_ff": args.d_ff,
            "rope_theta": args.rope_theta,
            "rms_norm_eps": args.rms_norm_eps,
        },
        "optimization": {
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "peak_lr": args.peak_lr,
            "final_lr_frac": args.final_lr_frac,
            "warmup_frac": args.warmup_frac,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "weight_decay": args.weight_decay,
            "adam_eps": args.adam_eps,
            "max_grad_norm": args.max_grad_norm,
            "precision": args.precision,
            "attention_backend": args.attention_backend,
            "seed": args.seed,
        },
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_final_checkpoint:
        checkpoint_path = args.output_dir / "final_checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "result": result,
            },
            checkpoint_path,
        )

    print("=" * 80)
    print("Training completed")
    print("=" * 80)
    print(f"Final validation loss: {final_validation_loss:.6f}")
    print(f"Training throughput:   {result['training_tokens_per_second']:,.0f} tok/s")
    print(f"Peak GPU memory:       {peak_memory_gib:.2f} GiB")
    print(f"Result:                {result_path}")


if __name__ == "__main__":
    main()
