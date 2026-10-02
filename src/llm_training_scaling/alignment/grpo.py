from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs separately, concatenate, shift and mask."""

    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt/output batch sizes differ: "
            f"{len(prompt_strs)} != {len(output_strs)}"
        )
    if not prompt_strs:
        raise ValueError("The batch must contain at least one example.")

    # The assignment explicitly requires no automatically added special tokens.
    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    output_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]

    combined_ids: list[list[int]] = []
    response_token_masks: list[list[bool]] = []

    for prompt, output in zip(prompt_ids, output_ids, strict=True):
        sequence = list(prompt) + list(output)

        if len(sequence) < 2:
            raise ValueError(
                "Each concatenated prompt/output sequence must contain "
                "at least two tokens."
            )

        combined_ids.append(sequence)

        # This mask is initially aligned with the unshifted complete sequence.
        response_token_masks.append(
            [False] * len(prompt) + [True] * len(output)
        )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

    batch_size = len(combined_ids)
    max_full_length = max(len(sequence) for sequence in combined_ids)

    padded_ids = torch.full(
        (batch_size, max_full_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    padded_response_mask = torch.zeros(
        (batch_size, max_full_length),
        dtype=torch.bool,
    )

    for row, (sequence, mask) in enumerate(
        zip(combined_ids, response_token_masks, strict=True)
    ):
        length = len(sequence)

        padded_ids[row, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
        )
        padded_response_mask[row, :length] = torch.tensor(
            mask,
            dtype=torch.bool,
        )

    # Causal LM alignment:
    # input_ids[t] predicts labels[t], which is original token t + 1.
    input_ids = padded_ids[:, :-1]
    labels = padded_ids[:, 1:]
    response_mask = padded_response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the selected next-token log-probability at each position."""

    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have the same shape, got "
            f"{tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )

    logits = model(input_ids=input_ids).logits

    # Shape: (batch_size, sequence_length, vocabulary_size)
    all_log_probs = F.log_softmax(logits, dim=-1)

    # Select log p(label_t | tokens before label_t).
    token_log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        "log_probs": token_log_probs,
    }

    if return_token_entropy:
        probabilities = all_log_probs.exp()
        token_entropy = -(probabilities * all_log_probs).sum(dim=-1)
        result["token_entropy"] = token_entropy

    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Grade every rollout and return raw rewards plus logging metadata."""

    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            f"response/ground-truth batch sizes differ: "
            f"{len(rollout_responses)} != {len(repeated_ground_truths)}"
        )
    if not rollout_responses:
        raise ValueError("The rollout batch must not be empty.")

    graded = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
            strict=True,
        )
    ]

    raw_rewards = torch.tensor(
        [item["reward"] for item in graded],
        dtype=torch.float32,
    )

    format_rewards = torch.tensor(
        [item["format_reward"] for item in graded],
        dtype=torch.float32,
    )

    answer_rewards = torch.tensor(
        [item["answer_reward"] for item in graded],
        dtype=torch.float32,
    )

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": format_rewards.mean().item(),
        "answer_reward_mean": answer_rewards.mean().item(),
    }

    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-group reward advantages."""

    if raw_rewards.ndim != 1:
        raise ValueError(
            f"raw_rewards must be one-dimensional, got {raw_rewards.shape}"
        )
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if raw_rewards.numel() % group_size != 0:
        raise ValueError(
            f"{raw_rewards.numel()} rewards cannot be divided into "
            f"groups of size {group_size}."
        )

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    if baseline == "mean":
        grouped_advantages = grouped_rewards - group_means
    elif baseline == "none":
        grouped_advantages = grouped_rewards.clone()
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    if advantage_normalizer == "std":
        # The assignment asks for torch.std's default sample standard deviation.
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = grouped_advantages / (
            group_stds + advantage_eps
        )
    elif advantage_normalizer == "mean":
        grouped_advantages = grouped_advantages / (
            group_means + advantage_eps
        )
    elif advantage_normalizer == "none":
        pass
    else:
        raise NotImplementedError(
            f"Unsupported advantage normalizer: {advantage_normalizer}"
        )

    advantages = grouped_advantages.reshape_as(raw_rewards)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_min": raw_rewards.min().item(),
        "reward_max": raw_rewards.max().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_min": advantages.min().item(),
        "advantage_max": advantages.max().item(),
    }

    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute an on-policy or importance-reweighted PG loss."""

    if policy_log_probs.ndim != 2:
        raise ValueError(
            "policy_log_probs must have shape "
            "(batch_size, sequence_length)."
        )

    advantages = raw_rewards_or_advantages

    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    elif advantages.ndim == 2 and advantages.shape[1] == 1:
        pass
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have shape "
            "(batch_size,) or (batch_size, 1)."
        )

    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            "Advantage batch size does not match "
            "log-probability batch size."
        )

    advantages = advantages.to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )

    if response_mask is not None:
        if response_mask.shape != policy_log_probs.shape:
            raise ValueError(
                "response_mask must match policy_log_probs."
            )
        response_mask = response_mask.to(
            device=policy_log_probs.device,
            dtype=torch.bool,
        )

    if importance_reweighting_method == "none":
        # On-policy REINFORCE/GRPO objective:
        #     A * log pi
        per_token_loss = -advantages * policy_log_probs

        return per_token_loss, {
            "clip_fraction": torch.zeros(
                (),
                device=policy_log_probs.device,
            ),
            "importance_ratio_mean": torch.ones(
                (),
                device=policy_log_probs.device,
            ),
        }

    if importance_reweighting_method not in {
        "noclip",
        "grpo",
        "gspo",
    }:
        raise NotImplementedError(
            "Unsupported importance reweighting method: "
            f"{importance_reweighting_method!r}"
        )

    if old_log_probs is None:
        raise ValueError(
            "old_log_probs is required for off-policy training."
        )
    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError(
            "old_log_probs must match policy_log_probs."
        )

    old_log_probs = old_log_probs.detach().to(
        device=policy_log_probs.device,
        dtype=policy_log_probs.dtype,
    )
    log_ratio = policy_log_probs - old_log_probs

    def masked_mean(values: torch.Tensor) -> torch.Tensor:
        if response_mask is None:
            return values.float().mean()

        mask_float = response_mask.to(values.dtype)
        return (
            (values * mask_float).sum()
            / mask_float.sum().clamp_min(1)
        ).float()

    if importance_reweighting_method in {
        "noclip",
        "grpo",
    }:
        # Token-level importance ratio:
        #     pi_theta(y_t | prefix) / pi_old(y_t | prefix)
        ratio = torch.exp(log_ratio)
        unclipped_objective = advantages * ratio

        if importance_reweighting_method == "noclip":
            per_token_loss = -unclipped_objective
            clip_mask = torch.zeros_like(
                ratio,
                dtype=torch.bool,
            )
        else:
            if cliprange is None or cliprange <= 0:
                raise ValueError(
                    "A positive cliprange is required for "
                    "clipped GRPO."
                )

            clipped_ratio = torch.clamp(
                ratio,
                min=1.0 - cliprange,
                max=1.0 + cliprange,
            )
            clipped_objective = advantages * clipped_ratio

            # PPO/GRPO clipped surrogate objective.
            objective = torch.minimum(
                unclipped_objective,
                clipped_objective,
            )
            per_token_loss = -objective
            clip_mask = (
                unclipped_objective > clipped_objective
            )

        return per_token_loss, {
            "clip_fraction": masked_mean(
                clip_mask.to(policy_log_probs.dtype)
            ).detach(),
            "importance_ratio_mean": masked_mean(
                ratio
            ).detach(),
        }

    # GSPO uses the geometric mean importance ratio over response
    # tokens, yielding one sequence-level ratio per rollout.
    if response_mask is None:
        raise ValueError(
            "response_mask is required for GSPO."
        )
    if cliprange is None or cliprange <= 0:
        raise ValueError(
            "A positive cliprange is required for GSPO."
        )

    mask_float = response_mask.to(policy_log_probs.dtype)
    response_lengths = mask_float.sum(
        dim=1,
    ).clamp_min(1)

    mean_log_ratio = (
        (log_ratio * mask_float).sum(dim=1)
        / response_lengths
    )
    sequence_ratio = torch.exp(mean_log_ratio)

    sequence_advantages = advantages.squeeze(1)
    unclipped_objective = (
        sequence_advantages * sequence_ratio
    )
    clipped_objective = (
        sequence_advantages
        * torch.clamp(
            sequence_ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )
    )

    sequence_objective = torch.minimum(
        unclipped_objective,
        clipped_objective,
    )
    sequence_clip_mask = (
        unclipped_objective > clipped_objective
    )

    # Repeating the sequence objective across token positions allows
    # the normal response mask/sequence aggregation path to be reused.
    per_token_loss = (
        -sequence_objective.unsqueeze(1)
    ).expand_as(policy_log_probs)

    return per_token_loss, {
        "clip_fraction": (
            sequence_clip_mask.float().mean().detach()
        ),
        "importance_ratio_mean": (
            sequence_ratio.float().mean().detach()
        ),
    }


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate token losses into one scalar microbatch loss."""

    if per_token_policy_gradient_loss.shape != mask.shape:
        raise ValueError(
            "Loss and mask must have identical shapes, got "
            f"{tuple(per_token_policy_gradient_loss.shape)} and "
            f"{tuple(mask.shape)}."
        )

    mask_float = mask.to(
        device=per_token_policy_gradient_loss.device,
        dtype=per_token_policy_gradient_loss.dtype,
    )
    masked_loss = per_token_policy_gradient_loss * mask_float

    if loss_normalization == "sequence":
        tokens_per_sequence = mask_float.sum(dim=1)

        if torch.any(tokens_per_sequence == 0):
            raise ValueError(
                "Every sequence must contain at least one response token."
            )

        loss_per_sequence = (
            masked_loss.sum(dim=1) / tokens_per_sequence
        )
        return loss_per_sequence.mean()

    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required for constant "
                "normalization."
            )
        if normalization_constant <= 0:
            raise ValueError(
                "normalization_constant must be positive."
            )

        return masked_loss.sum() / normalization_constant

    raise NotImplementedError(
        f"Unsupported loss normalization: {loss_normalization}"
    )


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal[
        "std", "none", "mean"
    ] = "std",
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal[
        "sequence", "constant"
    ] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute one GRPO-family optimizer update."""

    if baseline not in {"mean", "none"}:
        raise NotImplementedError(
            f"Unsupported baseline: {baseline!r}"
        )
    if advantage_normalizer not in {
        "std",
        "none",
        "mean",
    }:
        raise NotImplementedError(
            "Unsupported advantage normalizer: "
            f"{advantage_normalizer!r}"
        )
    if importance_reweighting_method not in {
        "none",
        "noclip",
        "grpo",
        "gspo",
    }:
        raise NotImplementedError(
            "Unsupported importance reweighting method: "
            f"{importance_reweighting_method!r}"
        )
    if loss_normalization not in {
        "sequence",
        "constant",
    }:
        raise NotImplementedError(
            "Unsupported loss normalization: "
            f"{loss_normalization!r}"
        )
    if (
        loss_normalization == "constant"
        and (
            normalization_constant is None
            or normalization_constant <= 0
        )
    ):
        raise ValueError(
            "A positive normalization_constant is required "
            "for constant loss normalization."
        )

    is_off_policy = (
        importance_reweighting_method != "none"
    )

    if is_off_policy and old_log_probs is None:
        raise ValueError(
            "old_log_probs is required for off-policy training."
        )
    if (
        importance_reweighting_method in {"grpo", "gspo"}
        and (cliprange is None or cliprange <= 0)
    ):
        raise ValueError(
            "A positive cliprange is required for clipped "
            "off-policy training."
        )

    batch_size = len(repeated_prompts)

    if not (
        batch_size
        == len(rollout_responses)
        == len(repeated_ground_truths)
    ):
        raise ValueError(
            "Prompts, responses and ground truths must have "
            "equal lengths."
        )
    if batch_size == 0:
        raise ValueError(
            "The rollout batch must not be empty."
        )
    if group_size <= 0 or batch_size % group_size != 0:
        raise ValueError(
            "group_size must be positive and divide "
            "the batch size."
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive."
        )
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "Batch size must be divisible by "
            "gradient_accumulation_steps."
        )

    if old_log_probs is not None:
        if old_log_probs.ndim != 2:
            raise ValueError(
                "old_log_probs must have shape "
                "(batch_size, sequence_length)."
            )
        if old_log_probs.shape[0] != batch_size:
            raise ValueError(
                "old_log_probs batch dimension does not "
                "match the rollout batch."
            )

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    advantages, advantage_metadata = (
        compute_group_normalized_rewards(
            raw_rewards=raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
    )
    advantages = advantages.reshape(-1)

    # Zero-advantage sequences have exactly zero contribution.
    active_indices = torch.nonzero(
        advantages != 0,
        as_tuple=False,
    ).flatten().tolist()
    active_count = len(active_indices)

    active_prompts = [
        repeated_prompts[index]
        for index in active_indices
    ]
    active_responses = [
        rollout_responses[index]
        for index in active_indices
    ]
    active_advantages = advantages[active_indices]

    device = next(model.parameters()).device
    microbatch_size = (
        batch_size // gradient_accumulation_steps
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    entropy_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    entropy_token_count = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    clip_fraction_weighted_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    importance_ratio_weighted_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    off_policy_weight_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )

    if active_count > 0:
        tokenized = tokenize_prompt_and_output(
            prompt_strs=active_prompts,
            output_strs=active_responses,
            tokenizer=tokenizer,
        )

        active_old_log_probs = None

        if old_log_probs is not None:
            active_index_tensor = torch.tensor(
                active_indices,
                device=old_log_probs.device,
                dtype=torch.long,
            )
            active_old_log_probs = old_log_probs.index_select(
                0,
                active_index_tensor,
            )

            active_sequence_length = tokenized[
                "input_ids"
            ].shape[1]

            if (
                active_old_log_probs.shape[1]
                < active_sequence_length
            ):
                raise ValueError(
                    "old_log_probs sequence dimension is "
                    "shorter than the tokenized active batch."
                )

            # Pruning can remove the longest sequence, so crop any
            # old-policy trailing padding columns.
            active_old_log_probs = active_old_log_probs[
                :, :active_sequence_length
            ]

        for start in range(
            0,
            active_count,
            microbatch_size,
        ):
            end = min(
                start + microbatch_size,
                active_count,
            )

            input_ids = tokenized["input_ids"][
                start:end
            ].to(device)
            labels = tokenized["labels"][
                start:end
            ].to(device)
            response_mask = tokenized["response_mask"][
                start:end
            ].to(device)
            microbatch_advantages = active_advantages[
                start:end
            ].to(device)

            scoring_output = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )

            microbatch_old_log_probs = None
            if active_old_log_probs is not None:
                microbatch_old_log_probs = (
                    active_old_log_probs[start:end].to(
                        device=device,
                        dtype=scoring_output[
                            "log_probs"
                        ].dtype,
                    )
                )

            per_token_loss, loss_metadata = (
                compute_policy_gradient_loss(
                    raw_rewards_or_advantages=(
                        microbatch_advantages
                    ),
                    policy_log_probs=(
                        scoring_output["log_probs"]
                    ),
                    importance_reweighting_method=(
                        importance_reweighting_method
                    ),
                    old_log_probs=(
                        microbatch_old_log_probs
                    ),
                    cliprange=cliprange,
                    response_mask=response_mask,
                )
            )

            microbatch_loss = (
                aggregate_loss_across_microbatch(
                    per_token_policy_gradient_loss=(
                        per_token_loss
                    ),
                    mask=response_mask,
                    loss_normalization=(
                        loss_normalization
                    ),
                    normalization_constant=(
                        normalization_constant
                    ),
                )
            )

            if loss_normalization == "sequence":
                backward_loss = microbatch_loss * (
                    (end - start) / batch_size
                )
            else:
                backward_loss = microbatch_loss

            backward_loss.backward()
            total_loss += (
                backward_loss.detach().float()
            )

            mask_float = response_mask.to(
                dtype=scoring_output[
                    "token_entropy"
                ].dtype
            )
            response_token_count = (
                mask_float.sum().detach().float()
            )

            entropy_sum += (
                scoring_output["token_entropy"]
                * mask_float
            ).sum().detach().float()
            entropy_token_count += response_token_count

            if is_off_policy:
                if (
                    importance_reweighting_method
                    == "gspo"
                ):
                    metadata_weight = torch.tensor(
                        end - start,
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    metadata_weight = response_token_count

                clip_fraction_weighted_sum += (
                    loss_metadata["clip_fraction"]
                    .detach()
                    .float()
                    * metadata_weight
                )
                importance_ratio_weighted_sum += (
                    loss_metadata[
                        "importance_ratio_mean"
                    ]
                    .detach()
                    .float()
                    * metadata_weight
                )
                off_policy_weight_sum += metadata_weight

    if max_grad_norm is not None:
        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
        )
    else:
        gradient_parts = [
            parameter.grad.detach().norm(2)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

        if gradient_parts:
            gradient_norm = torch.linalg.vector_norm(
                torch.stack(gradient_parts)
            )
        else:
            gradient_norm = torch.tensor(
                0.0,
                device=device,
            )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    token_entropy = (
        entropy_sum
        / entropy_token_count.clamp_min(1)
    )

    if is_off_policy:
        clip_fraction = (
            clip_fraction_weighted_sum
            / off_policy_weight_sum.clamp_min(1)
        )
        importance_ratio_mean = (
            importance_ratio_weighted_sum
            / off_policy_weight_sum.clamp_min(1)
        )
    else:
        clip_fraction = torch.zeros(
            (),
            device=device,
        )
        importance_ratio_mean = torch.ones(
            (),
            device=device,
        )

    metadata: dict[
        str,
        torch.Tensor | float,
    ] = {
        **reward_metadata,
        **advantage_metadata,
        "loss": total_loss.item(),
        "gradient_norm": gradient_norm.detach(),
        "token_entropy": token_entropy.detach(),
        "active_sequence_fraction": (
            active_count / batch_size
        ),
        "clip_fraction": clip_fraction.detach(),
        "importance_ratio_mean": (
            importance_ratio_mean.detach()
        ),
    }

    return total_loss.detach(), metadata
