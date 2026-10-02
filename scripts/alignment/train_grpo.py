from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from llm_training_scaling.alignment.rewards import r1_zero_reward_fn
from llm_training_scaling.alignment.grpo import (
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from llm_training_scaling.alignment.rollout_server import VLLMServer


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No examples found in {path}")

    return rows


def get_ground_truth(example: dict[str, Any]) -> str:
    return str(example["answer"]).split("####")[-1].strip()


def make_prompt(template: str, example: dict[str, Any]) -> str:
    return template.replace("{question}", str(example["question"]))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    return value


def take_training_examples(
    dataset: list[dict[str, Any]],
    order: list[int],
    cursor: int,
    count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []

    while len(selected) < count:
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0

        remaining = count - len(selected)
        available = len(order) - cursor
        take = min(remaining, available)

        selected.extend(
            dataset[index]
            for index in order[cursor : cursor + take]
        )
        cursor += take

    return selected, cursor


@torch.inference_mode()
def evaluate(
    server: VLLMServer,
    tokenizer,
    examples: list[dict[str, Any]],
    prompt_template: str,
    sampling_temperature: float,
    sampling_max_tokens: int,
    generation_batch_size: int,
    seed: int,
    output_path: Path,
    max_logged_examples: int = 16,
) -> dict[str, float]:
    prompts = [
        make_prompt(prompt_template, example)
        for example in examples
    ]
    ground_truths = [
        get_ground_truth(example)
        for example in examples
    ]

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": sampling_temperature,
            "max_tokens": sampling_max_tokens,
            "n": 1,
            "seed": seed,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
        batch_size=generation_batch_size,
    )

    if len(completions) != len(examples):
        raise RuntimeError(
            f"Expected {len(examples)} validation completions, "
            f"received {len(completions)}."
        )

    rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    response_lengths: list[int] = []

    for index, (example, ground_truth, completion) in enumerate(
        zip(examples, ground_truths, completions, strict=True)
    ):
        grade = r1_zero_reward_fn(completion.text, ground_truth)

        rewards.append(float(grade["reward"]))
        format_rewards.append(float(grade["format_reward"]))
        answer_rewards.append(float(grade["answer_reward"]))

        if completion.token_ids:
            response_length = len(completion.token_ids)
        else:
            response_length = len(
                tokenizer.encode(
                    completion.text,
                    add_special_tokens=False,
                )
            )

        response_lengths.append(response_length)

        if index < max_logged_examples:
            append_jsonl(
                output_path,
                {
                    "index": index,
                    "question": example["question"],
                    "ground_truth": ground_truth,
                    "response": completion.text,
                    "reward": grade["reward"],
                    "format_reward": grade["format_reward"],
                    "answer_reward": grade["answer_reward"],
                    "response_tokens": response_length,
                },
            )

    count = len(examples)

    return {
        "val_reward": sum(rewards) / count,
        "val_format_reward": sum(format_rewards) / count,
        "val_answer_reward": sum(answer_rewards) / count,
        "val_average_response_length": sum(response_lengths) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-0425-1B",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=ROOT / "src/llm_training_scaling/alignment/prompts/r1_zero.prompt",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=ROOT / "data/gsm8k/train.jsonl",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=ROOT / "data/gsm8k/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-val-examples", type=int, default=64)

    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--baseline",
        choices=["mean", "none"],
        default="mean",
    )
    parser.add_argument(
        "--advantage-normalizer",
        choices=["std", "none", "mean"],
        default="std",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=["sequence", "constant"],
        default="sequence",
    )
    parser.add_argument(
        "--normalization-constant",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--advantage-eps",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sampling-max-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--log-rollouts-every",
        type=int,
        default=1,
    )

    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--save-final",
        action="store_true",
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help=(
            "Responses per optimizer update. Defaults to "
            "rollout_batch_size."
        ),
    )
    parser.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default="none",
    )
    parser.add_argument(
        "--cliprange",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by group_size."
        )
    if (
        args.rollout_batch_size
        % args.gradient_accumulation_steps
        != 0
    ):
        raise ValueError(
            "rollout_batch_size must be divisible by "
            "gradient_accumulation_steps."
        )
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"Two GPUs are required; found {torch.cuda.device_count()}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "metrics.jsonl"
    rollout_path = args.output_dir / "train_rollouts.jsonl"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rng = random.Random(args.seed)

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    train_examples = read_jsonl(
        args.train_file,
        args.n_train_examples,
    )
    val_examples = read_jsonl(
        args.val_file,
        args.n_val_examples,
    )

    order = list(range(len(train_examples)))
    rng.shuffle(order)
    cursor = 0

    prompts_per_step = (
        args.rollout_batch_size // args.group_size
    )

    train_device = torch.device(f"cuda:{args.train_gpu}")
    torch.cuda.set_device(train_device)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading training policy on {train_device}")
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    )
    policy.config.use_cache = False
    policy.to(train_device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    server = VLLMServer(
        model_id=args.model,
        gpu=args.vllm_gpu,
        seed=args.seed,
        logging_level="ERROR",
        gpu_memory_utilization=(
            args.vllm_gpu_memory_utilization
        ),
        startup_timeout=args.startup_timeout,
    )

    try:
        print(f"Starting vLLM on cuda:{args.vllm_gpu}")
        server.start()

        print("Initializing NCCL weight synchronization")
        server.init_weight_sync(
            policy_device=str(train_device)
        )

        print("Synchronizing initial policy weights")
        server.sync_policy_weights(policy)

        print(
            f"Running initial validation on "
            f"{len(val_examples)} examples"
        )
        initial_val = evaluate(
            server=server,
            tokenizer=tokenizer,
            examples=val_examples,
            prompt_template=prompt_template,
            sampling_temperature=args.sampling_temperature,
            sampling_max_tokens=args.sampling_max_tokens,
            generation_batch_size=args.generation_batch_size,
            seed=args.seed,
            output_path=(
                args.output_dir / "val_step_0000.jsonl"
            ),
        )

        initial_row = {
            "type": "validation",
            "step": 0,
            **initial_val,
        }
        append_jsonl(metrics_path, initial_row)
        print(json.dumps(initial_row, indent=2))

        for step in range(1, args.num_steps + 1):
            step_start = time.perf_counter()

            batch_examples, cursor = take_training_examples(
                dataset=train_examples,
                order=order,
                cursor=cursor,
                count=prompts_per_step,
                rng=rng,
            )

            prompts = [
                make_prompt(prompt_template, example)
                for example in batch_examples
            ]
            ground_truths = [
                get_ground_truth(example)
                for example in batch_examples
            ]

            # The rollout policy must match the policy being updated.
            server.sync_policy_weights(policy)

            completions = server.generate_completions(
                prompts=prompts,
                sampling_params={
                    "temperature": args.sampling_temperature,
                    "max_tokens": args.sampling_max_tokens,
                    "n": args.group_size,
                    "seed": args.seed + step,
                    "stop": ["</answer>"],
                    "include_stop_str_in_output": True,
                },
                batch_size=args.generation_batch_size,
            )

            if len(completions) != args.rollout_batch_size:
                raise RuntimeError(
                    f"Expected {args.rollout_batch_size} rollouts, "
                    f"received {len(completions)}."
                )

            rollout_responses = [
                completion.text
                for completion in completions
            ]
            repeated_prompts = [
                prompt
                for prompt in prompts
                for _ in range(args.group_size)
            ]
            repeated_ground_truths = [
                ground_truth
                for ground_truth in ground_truths
                for _ in range(args.group_size)
            ]

            train_batch_size = (
                args.train_batch_size
                if args.train_batch_size is not None
                else args.rollout_batch_size
            )

            if train_batch_size <= 0:
                raise ValueError(
                    "train_batch_size must be positive."
                )
            if (
                args.rollout_batch_size
                % train_batch_size
                != 0
            ):
                raise ValueError(
                    "rollout_batch_size must be divisible "
                    "by train_batch_size."
                )
            if (
                train_batch_size
                % args.group_size
                != 0
            ):
                raise ValueError(
                    "train_batch_size must contain complete "
                    "reward groups."
                )
            if (
                train_batch_size
                % args.gradient_accumulation_steps
                != 0
            ):
                raise ValueError(
                    "train_batch_size must be divisible by "
                    "gradient_accumulation_steps."
                )
            if (
                args.importance_reweighting_method
                in {"grpo", "gspo"}
                and (
                    args.cliprange is None
                    or args.cliprange <= 0
                )
            ):
                raise ValueError(
                    "A positive cliprange is required for "
                    "clipped off-policy training."
                )

            policy_device = next(
                policy.parameters()
            ).device

            # Compute log-probabilities under the policy that
            # generated this rollout batch. They remain fixed while
            # the 32 optimizer updates make the current policy stale.
            old_log_probs = None

            if (
                args.importance_reweighting_method
                != "none"
            ):
                policy.eval()

                old_tokenized = (
                    tokenize_prompt_and_output(
                        prompt_strs=repeated_prompts,
                        output_strs=rollout_responses,
                        tokenizer=tokenizer,
                    )
                )

                old_log_prob_chunks = []

                with torch.inference_mode():
                    for old_start in range(
                        0,
                        args.rollout_batch_size,
                        train_batch_size,
                    ):
                        old_end = min(
                            old_start + train_batch_size,
                            args.rollout_batch_size,
                        )

                        old_scoring = (
                            get_response_log_probs(
                                model=policy,
                                input_ids=old_tokenized[
                                    "input_ids"
                                ][old_start:old_end].to(
                                    policy_device
                                ),
                                labels=old_tokenized[
                                    "labels"
                                ][old_start:old_end].to(
                                    policy_device
                                ),
                                return_token_entropy=False,
                            )
                        )

                        old_log_prob_chunks.append(
                            old_scoring[
                                "log_probs"
                            ].detach().cpu()
                        )

                old_log_probs = torch.cat(
                    old_log_prob_chunks,
                    dim=0,
                )

                del old_tokenized
                del old_log_prob_chunks

            num_optimizer_updates = (
                args.rollout_batch_size
                // train_batch_size
            )

            minibatch_losses = []
            minibatch_metadata = []

            for minibatch_index, minibatch_start in (
                enumerate(
                    range(
                        0,
                        args.rollout_batch_size,
                        train_batch_size,
                    )
                )
            ):
                minibatch_end = (
                    minibatch_start
                    + train_batch_size
                )
                update_start = time.perf_counter()

                minibatch_old_log_probs = None
                if old_log_probs is not None:
                    minibatch_old_log_probs = (
                        old_log_probs[
                            minibatch_start:
                            minibatch_end
                        ]
                    )

                minibatch_loss, current_metadata = (
                    grpo_train_step(
                        model=policy,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        gradient_accumulation_steps=(
                            args.gradient_accumulation_steps
                        ),
                        max_grad_norm=(
                            args.max_grad_norm
                        ),
                        reward_fn=r1_zero_reward_fn,
                        repeated_prompts=(
                            repeated_prompts[
                                minibatch_start:
                                minibatch_end
                            ]
                        ),
                        rollout_responses=(
                            rollout_responses[
                                minibatch_start:
                                minibatch_end
                            ]
                        ),
                        repeated_ground_truths=(
                            repeated_ground_truths[
                                minibatch_start:
                                minibatch_end
                            ]
                        ),
                        group_size=args.group_size,
                        baseline=args.baseline,
                        advantage_eps=(
                            args.advantage_eps
                        ),
                        advantage_normalizer=(
                            args.advantage_normalizer
                        ),
                        importance_reweighting_method=(
                            args.importance_reweighting_method
                        ),
                        old_log_probs=(
                            minibatch_old_log_probs
                        ),
                        cliprange=args.cliprange,
                        loss_normalization=(
                            args.loss_normalization
                        ),
                        normalization_constant=(
                            args.normalization_constant
                        ),
                    )
                )

                minibatch_losses.append(
                    float(
                        minibatch_loss
                        .detach()
                        .cpu()
                    )
                )
                minibatch_metadata.append(
                    current_metadata
                )

                # Preserve detailed within-rollout-batch behavior.
                # This lets us see clipping increase as the policy
                # becomes progressively staler.
                update_row = {
                    "type": "train_update",
                    "step": step,
                    "minibatch": (
                        minibatch_index + 1
                    ),
                    "optimizer_update": (
                        (step - 1)
                        * num_optimizer_updates
                        + minibatch_index
                        + 1
                    ),
                    "loss": minibatch_losses[-1],
                    "elapsed_seconds": (
                        time.perf_counter()
                        - update_start
                    ),
                    **{
                        key: scalar(value)
                        for key, value
                        in current_metadata.items()
                    },
                }
                append_jsonl(
                    metrics_path,
                    update_row,
                )

            # Aggregate the 32 optimizer-update metrics into one row
            # per rollout batch so validation curves remain directly
            # comparable with on-policy training.
            metadata = {}

            for key in minibatch_metadata[0]:
                values = [
                    float(scalar(item[key]))
                    for item in minibatch_metadata
                ]
                metadata[key] = (
                    sum(values) / len(values)
                )

            loss_value = (
                sum(minibatch_losses)
                / len(minibatch_losses)
            )
            loss = torch.tensor(
                loss_value,
                device=policy_device,
            )

            gradient_values = [
                float(
                    scalar(item["gradient_norm"])
                )
                for item in minibatch_metadata
            ]
            clip_values = [
                float(
                    scalar(
                        item.get(
                            "clip_fraction",
                            0.0,
                        )
                    )
                )
                for item in minibatch_metadata
            ]
            ratio_values = [
                float(
                    scalar(
                        item.get(
                            "importance_ratio_mean",
                            1.0,
                        )
                    )
                )
                for item in minibatch_metadata
            ]

            metadata.update(
                {
                    "gradient_norm_max": max(
                        gradient_values
                    ),
                    "clip_fraction_max": max(
                        clip_values
                    ),
                    "importance_ratio_min": min(
                        ratio_values
                    ),
                    "importance_ratio_max": max(
                        ratio_values
                    ),
                    "optimizer_updates": (
                        num_optimizer_updates
                    ),
                }
            )

            train_row = {
                "type": "train",
                "step": step,
                "loss": loss_value,
                "elapsed_seconds": (
                    time.perf_counter() - step_start
                ),
                **{
                    key: scalar(value)
                    for key, value in metadata.items()
                },
            }
            append_jsonl(metrics_path, train_row)

            print(
                f"step={step:03d} "
                f"loss={train_row['loss']:.6f} "
                f"reward={train_row['reward_mean']:.4f} "
                f"format={train_row['format_reward_mean']:.4f} "
                f"grad_norm={train_row['gradient_norm']:.4f} "
                f"clip={train_row.get('clip_fraction', 0.0):.4f} "
                f"clip_max={train_row.get('clip_fraction_max', 0.0):.4f} "
                f"ratio={train_row.get('importance_ratio_mean', 1.0):.4f} "
                f"updates={num_optimizer_updates} "
                f"time={train_row['elapsed_seconds']:.1f}s"
            )

            if (
                args.log_rollouts_every > 0
                and step % args.log_rollouts_every == 0
            ):
                for index in range(
                    min(8, len(rollout_responses))
                ):
                    grade = r1_zero_reward_fn(
                        rollout_responses[index],
                        repeated_ground_truths[index],
                    )
                    append_jsonl(
                        rollout_path,
                        {
                            "step": step,
                            "index": index,
                            "prompt": repeated_prompts[index],
                            "ground_truth": (
                                repeated_ground_truths[index]
                            ),
                            "response": rollout_responses[index],
                            **grade,
                        },
                    )

            should_evaluate = (
                step % args.eval_every == 0
                or step == args.num_steps
            )

            if should_evaluate:
                # grpo_train_step has updated the policy, so sync again.
                server.sync_policy_weights(policy)

                val_metrics = evaluate(
                    server=server,
                    tokenizer=tokenizer,
                    examples=val_examples,
                    prompt_template=prompt_template,
                    sampling_temperature=(
                        args.sampling_temperature
                    ),
                    sampling_max_tokens=(
                        args.sampling_max_tokens
                    ),
                    generation_batch_size=(
                        args.generation_batch_size
                    ),
                    seed=args.seed,
                    output_path=(
                        args.output_dir
                        / f"val_step_{step:04d}.jsonl"
                    ),
                )

                val_row = {
                    "type": "validation",
                    "step": step,
                    **val_metrics,
                }
                append_jsonl(metrics_path, val_row)
                print(json.dumps(val_row, indent=2))

        if args.save_final:
            final_dir = args.output_dir / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            print(f"Saved final model to {final_dir}")

    finally:
        server.stop()


if __name__ == "__main__":
    main()
