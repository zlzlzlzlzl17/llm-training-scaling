from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch
from torch import nn
from torch.optim import Optimizer


CheckpointDestination = (
    str
    | os.PathLike[str]
    | BinaryIO
    | IO[bytes]
)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: CheckpointDestination,
) -> None:
    """
    保存训练 checkpoint。

    checkpoint 包含：
        model_state_dict
        optimizer_state_dict
        iteration

    out 可以是文件路径，也可以是 BytesIO 等 binary file-like
    object。
    """

    if iteration < 0:
        raise ValueError(
            f"iteration must be non-negative, got {iteration}"
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointDestination,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    从 checkpoint 恢复模型、optimizer 和 iteration。

    Args:
        src:
            checkpoint 路径或 binary file-like object。

        model:
            已经创建好的、结构与 checkpoint 相同的模型。

        optimizer:
            已经创建好的 optimizer。

    Returns:
        保存 checkpoint 时的 iteration。
    """

    # 我们加载的是自己保存的完整训练 checkpoint。
    # weights_only=False 可明确允许 optimizer state 和普通
    # Python 数据结构。
    try:
        checkpoint = torch.load(
            src,
            weights_only=False,
        )
    except TypeError:
        # 兼容没有 weights_only 参数的较旧 PyTorch。
        checkpoint = torch.load(src)

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "iteration",
    }

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    iteration = checkpoint["iteration"]

    if not isinstance(iteration, int):
        raise TypeError(
            "Checkpoint iteration must be an integer, "
            f"got {type(iteration).__name__}"
        )

    return iteration
