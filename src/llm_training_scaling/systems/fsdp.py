from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from llm_training_scaling.systems.benchmark_model import Embedding, Linear


def _make_local_shard(
    full_weight: torch.Tensor,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, int]:
    flat = full_weight.detach().contiguous().view(-1)
    shard_numel = math.ceil(flat.numel() / world_size)
    padded_numel = shard_numel * world_size

    if padded_numel > flat.numel():
        padding = torch.zeros(
            padded_numel - flat.numel(),
            dtype=flat.dtype,
            device=flat.device,
        )
        flat = torch.cat((flat, padding))

    start = rank * shard_numel
    local_shard = flat.narrow(
        0,
        start,
        shard_numel,
    ).clone()

    return local_shard, shard_numel


def _all_gather_full_weight(
    local_weight: torch.Tensor,
    full_numel: int,
    full_shape: tuple[int, ...],
    compute_dtype: torch.dtype | None,
    world_size: int,
) -> torch.Tensor:
    communication_shard = (
        local_weight.to(compute_dtype)
        if compute_dtype is not None
        else local_weight
    )

    if world_size == 1:
        gathered_flat = communication_shard
    else:
        gathered = [
            torch.empty_like(communication_shard)
            for _ in range(world_size)
        ]

        dist.all_gather(
            gathered,
            communication_shard,
        )

        gathered_flat = torch.cat(gathered)

    return (
        gathered_flat[:full_numel]
        .contiguous()
        .view(full_shape)
    )


def _reduce_scatter_gradient(
    full_gradient: torch.Tensor,
    local_weight: torch.Tensor,
    shard_numel: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    flat_gradient = full_gradient.contiguous().view(-1)
    padded_numel = shard_numel * world_size

    if flat_gradient.numel() < padded_numel:
        padding = torch.zeros(
            padded_numel - flat_gradient.numel(),
            dtype=flat_gradient.dtype,
            device=flat_gradient.device,
        )
        flat_gradient = torch.cat(
            (flat_gradient, padding)
        )

    if world_size == 1:
        local_gradient = flat_gradient[:shard_numel]
        return local_gradient.to(local_weight.dtype)

    backend = str(dist.get_backend()).lower()

    if "nccl" in backend:
        # GPU benchmark path: actual reduce-scatter.
        local_gradient = torch.empty(
            shard_numel,
            dtype=flat_gradient.dtype,
            device=flat_gradient.device,
        )

        dist.reduce_scatter_tensor(
            local_gradient,
            flat_gradient,
            op=dist.ReduceOp.SUM,
        )

        local_gradient.div_(world_size)

        return local_gradient.to(local_weight.dtype)

    # Gloo test path. Some Gloo builds do not implement
    # reduce_scatter for every device/dtype combination.
    reduced = flat_gradient.to(local_weight.dtype)

    dist.all_reduce(
        reduced,
        op=dist.ReduceOp.SUM,
    )

    start = rank * shard_numel

    return (
        reduced.narrow(0, start, shard_numel)
        .clone()
        .div_(world_size)
    )


class _ShardedLinearFunction(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        local_weight: torch.Tensor,
        full_shape: tuple[int, ...],
        full_numel: int,
        shard_numel: int,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        full_weight = _all_gather_full_weight(
            local_weight=local_weight,
            full_numel=full_numel,
            full_shape=full_shape,
            compute_dtype=compute_dtype,
            world_size=world_size,
        )

        compute_inputs = inputs.to(
            full_weight.dtype
        )

        output = torch.matmul(
            compute_inputs,
            full_weight.transpose(0, 1),
        )

        ctx.save_for_backward(
            inputs,
            local_weight,
        )
        ctx.full_shape = full_shape
        ctx.full_numel = full_numel
        ctx.shard_numel = shard_numel
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.compute_dtype = compute_dtype

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        inputs, local_weight = (
            ctx.saved_tensors
        )

        # Linear backward needs the full weight to compute
        # the input gradient.
        full_weight = _all_gather_full_weight(
            local_weight=local_weight,
            full_numel=ctx.full_numel,
            full_shape=ctx.full_shape,
            compute_dtype=ctx.compute_dtype,
            world_size=ctx.world_size,
        )

        compute_inputs = inputs.to(
            full_weight.dtype
        )
        compute_grad_output = grad_output.to(
            full_weight.dtype
        )

        grad_inputs = torch.matmul(
            compute_grad_output,
            full_weight,
        ).to(inputs.dtype)

        flattened_inputs = (
            compute_inputs.reshape(
                -1,
                ctx.full_shape[1],
            )
        )
        flattened_grad_output = (
            compute_grad_output.reshape(
                -1,
                ctx.full_shape[0],
            )
        )

        full_grad_weight = (
            flattened_grad_output.transpose(0, 1)
            @ flattened_inputs
        )

        local_grad_weight = (
            _reduce_scatter_gradient(
                full_gradient=full_grad_weight,
                local_weight=local_weight,
                shard_numel=ctx.shard_numel,
                rank=ctx.rank,
                world_size=ctx.world_size,
            )
        )

        return (
            grad_inputs,
            local_grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _ShardedEmbeddingFunction(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        token_ids: torch.Tensor,
        local_weight: torch.Tensor,
        full_shape: tuple[int, ...],
        full_numel: int,
        shard_numel: int,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        full_weight = _all_gather_full_weight(
            local_weight=local_weight,
            full_numel=full_numel,
            full_shape=full_shape,
            compute_dtype=compute_dtype,
            world_size=world_size,
        )

        output = F.embedding(
            token_ids,
            full_weight,
        )

        ctx.save_for_backward(
            token_ids,
            local_weight,
        )
        ctx.full_shape = full_shape
        ctx.full_numel = full_numel
        ctx.shard_numel = shard_numel
        ctx.rank = rank
        ctx.world_size = world_size

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        token_ids, local_weight = (
            ctx.saved_tensors
        )

        # Use PyTorch's embedding backward so deterministic
        # algorithm settings are respected.
        with torch.enable_grad():
            dummy_weight = torch.zeros(
                ctx.full_shape,
                dtype=grad_output.dtype,
                device=grad_output.device,
                requires_grad=True,
            )

            dummy_output = F.embedding(
                token_ids,
                dummy_weight,
            )

            (full_grad_weight,) = (
                torch.autograd.grad(
                    outputs=dummy_output,
                    inputs=dummy_weight,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                )
            )

        local_grad_weight = (
            _reduce_scatter_gradient(
                full_gradient=full_grad_weight,
                local_weight=local_weight,
                shard_numel=ctx.shard_numel,
                rank=ctx.rank,
                world_size=ctx.world_size,
            )
        )

        return (
            None,
            local_grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FSDPLinear(Linear):
    def __init__(
        self,
        original: Linear,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> None:
        nn.Module.__init__(self)

        full_weight = original.weight.detach()

        self._fsdp_full_shape = tuple(
            full_weight.shape
        )
        self._fsdp_full_numel = (
            full_weight.numel()
        )
        self._fsdp_rank = rank
        self._fsdp_world_size = world_size
        self._fsdp_compute_dtype = (
            compute_dtype
        )

        local_shard, shard_numel = (
            _make_local_shard(
                full_weight,
                rank,
                world_size,
            )
        )

        self._fsdp_shard_numel = shard_numel

        self.weight = nn.Parameter(
            local_shard,
            requires_grad=(
                original.weight.requires_grad
            ),
        )

        self.train(original.training)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return _ShardedLinearFunction.apply(
            inputs,
            self.weight,
            self._fsdp_full_shape,
            self._fsdp_full_numel,
            self._fsdp_shard_numel,
            self._fsdp_rank,
            self._fsdp_world_size,
            self._fsdp_compute_dtype,
        )

    def gather_full_weight(self) -> torch.Tensor:
        return _all_gather_full_weight(
            local_weight=self.weight.detach(),
            full_numel=self._fsdp_full_numel,
            full_shape=self._fsdp_full_shape,
            compute_dtype=None,
            world_size=self._fsdp_world_size,
        )

    def extra_repr(self) -> str:
        return (
            f"full_shape={self._fsdp_full_shape}, "
            f"local_shard={self.weight.numel()}"
        )


class FSDPEmbedding(Embedding):
    def __init__(
        self,
        original: Embedding,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> None:
        nn.Module.__init__(self)

        full_weight = original.weight.detach()

        self._fsdp_full_shape = tuple(
            full_weight.shape
        )
        self._fsdp_full_numel = (
            full_weight.numel()
        )
        self._fsdp_rank = rank
        self._fsdp_world_size = world_size
        self._fsdp_compute_dtype = (
            compute_dtype
        )

        local_shard, shard_numel = (
            _make_local_shard(
                full_weight,
                rank,
                world_size,
            )
        )

        self._fsdp_shard_numel = shard_numel

        self.weight = nn.Parameter(
            local_shard,
            requires_grad=(
                original.weight.requires_grad
            ),
        )

        self.train(original.training)

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        return _ShardedEmbeddingFunction.apply(
            token_ids,
            self.weight,
            self._fsdp_full_shape,
            self._fsdp_full_numel,
            self._fsdp_shard_numel,
            self._fsdp_rank,
            self._fsdp_world_size,
            self._fsdp_compute_dtype,
        )

    def gather_full_weight(self) -> torch.Tensor:
        return _all_gather_full_weight(
            local_weight=self.weight.detach(),
            full_numel=self._fsdp_full_numel,
            full_shape=self._fsdp_full_shape,
            compute_dtype=None,
            world_size=self._fsdp_world_size,
        )

    def extra_repr(self) -> str:
        return (
            f"full_shape={self._fsdp_full_shape}, "
            f"local_shard={self.weight.numel()}"
        )


class FullyShardedDataParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is unavailable"
            )

        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize the process group before FSDP"
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype

        # Begin from identical full parameters.
        with torch.no_grad():
            for parameter in module.parameters():
                dist.broadcast(
                    parameter.data,
                    src=0,
                )

        self.module = module

        self._replace_sharded_layers(
            self.module
        )

        sharded_parameter_ids = {
            id(submodule.weight)
            for submodule
            in self.module.modules()
            if isinstance(
                submodule,
                (FSDPLinear, FSDPEmbedding),
            )
        }

        self._replicated_parameters = [
            parameter
            for parameter
            in self.module.parameters()
            if id(parameter)
            not in sharded_parameter_ids
        ]

    def _replace_sharded_layers(
        self,
        parent: nn.Module,
    ) -> None:
        for name, child in list(
            parent.named_children()
        ):
            if isinstance(
                child,
                (FSDPLinear, FSDPEmbedding),
            ):
                continue

            if isinstance(child, Linear):
                replacement = FSDPLinear(
                    original=child,
                    rank=self.rank,
                    world_size=self.world_size,
                    compute_dtype=self.compute_dtype,
                )
                setattr(parent, name, replacement)

            elif isinstance(child, Embedding):
                replacement = FSDPEmbedding(
                    original=child,
                    rank=self.rank,
                    world_size=self.world_size,
                    compute_dtype=self.compute_dtype,
                )
                setattr(parent, name, replacement)

            else:
                self._replace_sharded_layers(
                    child
                )

    def forward(
        self,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if self.compute_dtype is None:
            return self.module(*inputs, **kwargs)

        try:
            parameter = next(self.module.parameters())
        except StopIteration:
            return self.module(*inputs, **kwargs)

        # CUDA autocast handles mixed dtypes in operations outside
        # Linear/Embedding, such as RoPE and attention einsums.
        # CPU tests continue using the explicit weight-casting path.
        if parameter.device.type != "cuda":
            return self.module(*inputs, **kwargs)

        with torch.autocast(
            device_type="cuda",
            dtype=self.compute_dtype,
        ):
            return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(
        self,
    ) -> None:
        # Linear and Embedding gradients were synchronized
        # by reduce-scatter during backward. Norms and any
        # other small parameters remain replicated.
        for parameter in (
            self._replicated_parameters
        ):
            if parameter.grad is None:
                continue

            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
            )

            parameter.grad.div_(
                self.world_size
            )

    @torch.no_grad()
    def gather_full_params(
        self,
    ) -> dict[str, torch.Tensor]:
        modules = dict(
            self.module.named_modules()
        )

        full_parameters: dict[
            str,
            torch.Tensor,
        ] = {}

        for name, parameter in (
            self.module.named_parameters()
        ):
            parent_name, _, attribute = (
                name.rpartition(".")
            )

            parent = (
                modules[parent_name]
                if parent_name
                else self.module
            )

            if (
                attribute == "weight"
                and isinstance(
                    parent,
                    (
                        FSDPLinear,
                        FSDPEmbedding,
                    ),
                )
            ):
                full_parameters[name] = (
                    parent.gather_full_weight()
                )
            else:
                full_parameters[name] = (
                    parameter.detach().clone()
                )

        return full_parameters
