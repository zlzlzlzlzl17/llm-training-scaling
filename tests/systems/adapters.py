from __future__ import annotations

import torch



def get_flashattention_autograd_function_pytorch() -> type:
    """Return the pure-PyTorch FlashAttention autograd function."""
    from llm_training_scaling.systems.flash_attention import FlashAttentionPytorch

    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> type:
    """Return the Triton FlashAttention autograd function."""
    from llm_training_scaling.systems.flash_attention_triton import FlashAttentionTriton

    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """Return the current DDP implementation."""
    from llm_training_scaling.systems.ddp import OverlapDDP

    return OverlapDDP(module)


def ddp_on_after_backward(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """Synchronize gradients before the optimizer step."""
    del optimizer
    ddp_model.finish_gradient_synchronization()



def get_fsdp(
    module: torch.nn.Module,
    compute_dtype: torch.dtype | None = None,
) -> torch.nn.Module:
    from llm_training_scaling.systems.fsdp_prefetch import (
        PrefetchFullyShardedDataParallel,
    )

    return PrefetchFullyShardedDataParallel(
        module=module,
        compute_dtype=compute_dtype,
    )


def fsdp_on_after_backward(
    fsdp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    del optimizer
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(
    fsdp_model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return fsdp_model.gather_full_params()

def get_sharded_optimizer(
    params,
    optimizer_cls: type[torch.optim.Optimizer],
    **kwargs,
) -> torch.optim.Optimizer:
    """Construct the optimizer-state-sharded optimizer."""
    from llm_training_scaling.systems.sharded_optimizer import (
        ShardedOptimizer,
    )

    return ShardedOptimizer(
        params=params,
        optimizer_cls=optimizer_cls,
        **kwargs,
    )
