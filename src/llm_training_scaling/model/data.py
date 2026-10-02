from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token-ID 序列中随机采样 language-modeling batch。

    Args:
        dataset:
            一维 NumPy array 或 np.memmap，包含连续 token IDs。

        batch_size:
            每个 batch 中的序列数量。

        context_length:
            每条输入序列的 token 数量。

        device:
            返回 tensor 所在设备，例如 "cpu"、"cuda" 或 "cuda:0"。

    Returns:
        inputs:
            shape = (batch_size, context_length)

        targets:
            shape = (batch_size, context_length)

            targets 是 inputs 向右移动一个 token 后的结果。
    """

    if dataset.ndim != 1:
        raise ValueError(
            "dataset must be a one-dimensional token sequence, "
            f"got shape {dataset.shape}"
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )

    if context_length <= 0:
        raise ValueError(
            f"context_length must be positive, got {context_length}"
        )

    # 对于起点 start，需要访问：
    #
    # dataset[start : start + context_length + 1]
    #
    # 因此至少需要 context_length + 1 个 token。
    if len(dataset) <= context_length:
        raise ValueError(
            "dataset must contain more tokens than context_length: "
            f"dataset length={len(dataset)}, "
            f"context_length={context_length}"
        )

    # 有效起点：
    #
    # 0, 1, ..., len(dataset) - context_length - 1
    #
    # np.random.randint 的 high 参数不包含在采样范围内。
    starting_indices = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size,
    )

    # offsets:
    # [0, 1, ..., context_length - 1]
    offsets = np.arange(
        context_length,
        dtype=np.int64,
    )

    # token_indices shape:
    # (batch_size, context_length)
    token_indices = (
        starting_indices[:, None]
        + offsets[None, :]
    )

    # 使用 advanced indexing 从 NumPy array 或 memmap 中
    # 只取出当前 batch 所需的数据。
    input_array = np.asarray(
        dataset[token_indices],
        dtype=np.int64,
    )

    target_array = np.asarray(
        dataset[token_indices + 1],
        dtype=np.int64,
    )

    # torch.from_numpy 不复制底层数组；
    # 然后统一移动到请求的 device。
    inputs = torch.from_numpy(input_array).to(
        device=device,
        dtype=torch.long,
    )

    targets = torch.from_numpy(target_array).to(
        device=device,
        dtype=torch.long,
    )

    return inputs, targets
