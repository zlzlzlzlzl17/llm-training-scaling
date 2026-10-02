from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer。

    对每个参数 theta 保存：

        m：梯度的一阶矩估计
        v：梯度平方的二阶矩估计
        step：该参数已经更新的次数

    更新过程：

        theta <- theta - lr * weight_decay * theta

        m <- beta1 * m + (1 - beta1) * grad
        v <- beta2 * v + (1 - beta2) * grad^2

        adjusted_lr =
            lr * sqrt(1 - beta2^t) / (1 - beta1^t)

        theta <- theta
                 - adjusted_lr
                 * m / (sqrt(v) + eps)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter]
        | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        if eps < 0:
            raise ValueError(
                f"Invalid epsilon value: {eps}"
            )

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )

        if len(betas) != 2:
            raise ValueError(
                "betas must contain exactly two values."
            )

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError(
                f"Invalid beta1 value: {beta1}"
            )

        if not 0 <= beta2 < 1:
            raise ValueError(
                f"Invalid beta2 value: {beta2}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        """
        执行一次 AdamW 参数更新。

        closure 是 PyTorch Optimizer API 的可选接口，
        可用于重新计算 loss。
        """

        loss: torch.Tensor | None = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 不同 parameter group 可以使用不同的超参数。
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]

            for parameter in group["params"]:
                gradient = parameter.grad

                # 没有梯度的参数不更新，也不创建 optimizer state。
                if gradient is None:
                    continue

                if gradient.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients."
                    )

                state = self.state[parameter]

                # 第一次遇到该参数时初始化状态。
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                # 作业中 t 从 1 开始。
                state["step"] += 1
                step: int = state["step"]

                # --------------------------------------------------
                # 1. Decoupled weight decay
                #
                # theta <- theta - lr * lambda * theta
                #       = theta * (1 - lr * lambda)
                # --------------------------------------------------

                if weight_decay != 0:
                    parameter.mul_(
                        1.0 - lr * weight_decay
                    )

                # --------------------------------------------------
                # 2. 更新一阶矩
                #
                # m <- beta1*m + (1-beta1)*g
                # --------------------------------------------------

                exp_avg.mul_(beta1).add_(
                    gradient,
                    alpha=1.0 - beta1,
                )

                # --------------------------------------------------
                # 3. 更新二阶矩
                #
                # v <- beta2*v + (1-beta2)*g^2
                # --------------------------------------------------

                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                # --------------------------------------------------
                # 4. Bias correction
                #
                # adjusted_lr =
                #     lr * sqrt(1-beta2^t)/(1-beta1^t)
                # --------------------------------------------------

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                adjusted_learning_rate = (
                    lr
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                # --------------------------------------------------
                # 5. 参数更新
                #
                # theta <- theta
                #          - adjusted_lr*m/(sqrt(v)+eps)
                # --------------------------------------------------

                denominator = (
                    exp_avg_sq.sqrt()
                    .add_(eps)
                )

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-adjusted_learning_rate,
                )

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    带 linear warmup 的 cosine learning-rate schedule。

    阶段一：linear warmup
        0 <= it < warmup_iters

    阶段二：cosine annealing
        warmup_iters <= it <= cosine_cycle_iters

    阶段三：post-annealing
        it > cosine_cycle_iters
    """

    if it < 0:
        raise ValueError(
            f"it must be non-negative, got {it}"
        )

    if max_learning_rate < 0:
        raise ValueError(
            "max_learning_rate must be non-negative, "
            f"got {max_learning_rate}"
        )

    if min_learning_rate < 0:
        raise ValueError(
            "min_learning_rate must be non-negative, "
            f"got {min_learning_rate}"
        )

    if min_learning_rate > max_learning_rate:
        raise ValueError(
            "min_learning_rate cannot exceed "
            "max_learning_rate."
        )

    if warmup_iters < 0:
        raise ValueError(
            "warmup_iters must be non-negative, "
            f"got {warmup_iters}"
        )

    if cosine_cycle_iters < warmup_iters:
        raise ValueError(
            "cosine_cycle_iters must be greater than or "
            "equal to warmup_iters."
        )

    # ----------------------------------------------------------
    # 1. Linear warmup
    #
    # lr = (it / warmup_iters) * max_learning_rate
    #
    # it = 0 时，lr = 0
    # it 接近 warmup_iters 时，lr 接近 max_learning_rate
    # ----------------------------------------------------------

    if it < warmup_iters:
        return (
            it
            / warmup_iters
            * max_learning_rate
        )

    # ----------------------------------------------------------
    # 2. Post-annealing
    #
    # 超过 cosine cycle 终点后，保持 minimum LR。
    # ----------------------------------------------------------

    if it > cosine_cycle_iters:
        return min_learning_rate

    # ----------------------------------------------------------
    # 3. Cosine annealing
    #
    # progress:
    #   it = warmup_iters       -> 0
    #   it = cosine_cycle_iters -> 1
    #
    # cosine:
    #   progress = 0 -> cos(0)  = 1
    #   progress = 1 -> cos(pi) = -1
    # ----------------------------------------------------------

    # 特殊情况：没有 cosine decay 区间。
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    progress = (
        (it - warmup_iters)
        / (cosine_cycle_iters - warmup_iters)
    )

    cosine_factor = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return (
        min_learning_rate
        + cosine_factor
        * (max_learning_rate - min_learning_rate)
    )

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    对一组参数的梯度执行 global L2-norm clipping。

    所有参数梯度被视为拼接成一个长向量 g：

        ||g||_2 = sqrt(sum_i sum_j grad_i[j]^2)

    如果：

        ||g||_2 <= max_l2_norm

    则不修改梯度。

    如果：

        ||g||_2 > max_l2_norm

    则所有梯度统一乘以：

        max_l2_norm / (||g||_2 + 1e-6)

    函数原地修改 parameter.grad，不返回新梯度。
    """

    if max_l2_norm < 0:
        raise ValueError(
            "max_l2_norm must be non-negative, "
            f"got {max_l2_norm}"
        )

    # parameters 可能是 model.parameters() 这样的 generator，
    # 所以先收集有梯度的参数，避免 generator 被重复消费。
    parameters_with_grad = [
        parameter
        for parameter in parameters
        if parameter.grad is not None
    ]

    # 没有梯度时无需处理。
    if not parameters_with_grad:
        return

    # 使用 float32 计算 norm，避免 float16/bfloat16 平方溢出。
    #
    # total_squared_norm =
    #     sum over all parameters and all gradient elements of g^2
    first_gradient = parameters_with_grad[0].grad

    assert first_gradient is not None

    total_squared_norm = torch.zeros(
        (),
        device=first_gradient.device,
        dtype=torch.float32,
    )

    for parameter in parameters_with_grad:
        gradient = parameter.grad

        assert gradient is not None

        if gradient.is_sparse:
            raise RuntimeError(
                "gradient_clipping does not support sparse gradients."
            )

        gradient_float = gradient.detach().to(
            device=total_squared_norm.device,
            dtype=torch.float32,
        )

        total_squared_norm += torch.sum(
            gradient_float * gradient_float
        )

    total_l2_norm = torch.sqrt(total_squared_norm)

    # 只有超过最大 norm 时才缩放。
    if total_l2_norm > max_l2_norm:
        clip_coefficient = (
            max_l2_norm
            / (total_l2_norm + 1e-6)
        )

        with torch.no_grad():
            for parameter in parameters_with_grad:
                gradient = parameter.grad

                assert gradient is not None

                gradient.mul_(
                    clip_coefficient.to(
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                )

