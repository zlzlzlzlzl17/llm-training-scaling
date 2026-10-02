from __future__ import annotations

import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    计算平均 Cross-Entropy Loss。

    inputs:
        (..., vocab_size)

        每个位置对应整个 vocabulary 的未归一化 logits。

    targets:
        (...)

        每个位置的正确类别/token ID。

    返回：
        标量 tensor，表示所有位置的平均 loss。

    对单个样本：

        loss = log(sum_j exp(logit_j)) - logit_target

    为了数值稳定，使用：

        m = max(logits)

        log(sum_j exp(logit_j))
        = m + log(sum_j exp(logit_j - m))
    """

    if inputs.ndim < 1:
        raise ValueError(
            "inputs must have at least one dimension."
        )

    if targets.shape != inputs.shape[:-1]:
        raise ValueError(
            "targets shape must equal inputs.shape[:-1]: "
            f"inputs shape={tuple(inputs.shape)}, "
            f"targets shape={tuple(targets.shape)}"
        )

    if inputs.shape[-1] <= 0:
        raise ValueError(
            "The vocabulary dimension must be positive."
        )

    # gather() 要求索引使用整数类型。
    targets = targets.to(
        device=inputs.device,
        dtype=torch.long,
    )

    # ----------------------------------------------------------
    # 1. Numerically stable log-sum-exp
    #
    # maximum_logits:
    #     (..., 1)
    # ----------------------------------------------------------

    maximum_logits = torch.max(
        inputs,
        dim=-1,
        keepdim=True,
    ).values

    shifted_logits = inputs - maximum_logits

    log_sum_exp = (
        maximum_logits.squeeze(-1)
        + torch.log(
            torch.sum(
                torch.exp(shifted_logits),
                dim=-1,
            )
        )
    )

    # ----------------------------------------------------------
    # 2. 找出每个样本正确类别对应的 logit
    #
    # inputs:
    #     (..., vocab_size)
    #
    # targets.unsqueeze(-1):
    #     (..., 1)
    #
    # target_logits:
    #     (...)
    # ----------------------------------------------------------

    target_logits = torch.gather(
        inputs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # ----------------------------------------------------------
    # 3. 每个位置的 Cross-Entropy
    #
    # CE = logsumexp(logits) - correct_class_logit
    # ----------------------------------------------------------

    losses = log_sum_exp - target_logits

    # 对 batch 和其他所有 leading dimensions 取平均。
    return losses.mean()
