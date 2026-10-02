from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from llm_training_scaling.systems.fsdp import (
    FSDPEmbedding,
    FSDPLinear,
    FullyShardedDataParallel,
    _all_gather_full_weight,
    _reduce_scatter_gradient,
)


class _PrefetchStateMixin:
    """Asynchronous forward weight all-gather state."""

    def _initialize_prefetch_state(self) -> None:
        self._prefetch_work: dist.Work | None = None
        self._prefetch_input: torch.Tensor | None = None
        self._prefetched_flat: torch.Tensor | None = None

    @torch.no_grad()
    def start_weight_prefetch(self) -> None:
        if self._prefetch_work is not None:
            return

        if self._prefetched_flat is not None:
            return

        if self._fsdp_world_size == 1:
            return

        # Keep CPU/Gloo tests on the existing synchronous path.
        if self.weight.device.type != "cuda":
            return

        communication_shard = self.weight.detach()

        if self._fsdp_compute_dtype is not None:
            communication_shard = communication_shard.to(
                self._fsdp_compute_dtype
            )

        communication_shard = communication_shard.contiguous()

        gathered_flat = torch.empty(
            communication_shard.numel()
            * self._fsdp_world_size,
            dtype=communication_shard.dtype,
            device=communication_shard.device,
        )

        work = dist.all_gather_into_tensor(
            gathered_flat,
            communication_shard,
            async_op=True,
        )

        # Keep the input/output tensors alive until NCCL completes.
        self._prefetch_input = communication_shard
        self._prefetched_flat = gathered_flat
        self._prefetch_work = work

    @torch.no_grad()
    def consume_full_weight(self) -> torch.Tensor:
        if (
            self._prefetch_work is None
            or self._prefetched_flat is None
        ):
            return _all_gather_full_weight(
                local_weight=self.weight,
                full_numel=self._fsdp_full_numel,
                full_shape=self._fsdp_full_shape,
                compute_dtype=self._fsdp_compute_dtype,
                world_size=self._fsdp_world_size,
            )

        self._prefetch_work.wait()

        gathered_flat = self._prefetched_flat

        full_weight = (
            gathered_flat[: self._fsdp_full_numel]
            .contiguous()
            .view(self._fsdp_full_shape)
        )

        # full_weight retains the gathered storage through its view.
        self._prefetch_work = None
        self._prefetch_input = None
        self._prefetched_flat = None

        return full_weight

    @torch.no_grad()
    def discard_unconsumed_prefetch(self) -> None:
        if self._prefetch_work is not None:
            self._prefetch_work.wait()

        self._prefetch_work = None
        self._prefetch_input = None
        self._prefetched_flat = None


class _PrefetchedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        local_weight: torch.Tensor,
        full_weight: torch.Tensor,
        full_shape: tuple[int, ...],
        full_numel: int,
        shard_numel: int,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        compute_inputs = inputs.to(full_weight.dtype)

        output = torch.matmul(
            compute_inputs,
            full_weight.transpose(0, 1),
        )

        ctx.save_for_backward(inputs, local_weight)
        ctx.full_shape = full_shape
        ctx.full_numel = full_numel
        ctx.shard_numel = shard_numel
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.compute_dtype = compute_dtype

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, local_weight = ctx.saved_tensors

        # Backward still gathers synchronously. The assignment's
        # two-layer lookahead requirement applies to forward.
        full_weight = _all_gather_full_weight(
            local_weight=local_weight,
            full_numel=ctx.full_numel,
            full_shape=ctx.full_shape,
            compute_dtype=ctx.compute_dtype,
            world_size=ctx.world_size,
        )

        compute_inputs = inputs.to(full_weight.dtype)
        compute_grad_output = grad_output.to(
            full_weight.dtype
        )

        grad_inputs = torch.matmul(
            compute_grad_output,
            full_weight,
        ).to(inputs.dtype)

        flattened_inputs = compute_inputs.reshape(
            -1,
            ctx.full_shape[1],
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

        local_grad_weight = _reduce_scatter_gradient(
            full_gradient=full_grad_weight,
            local_weight=local_weight,
            shard_numel=ctx.shard_numel,
            rank=ctx.rank,
            world_size=ctx.world_size,
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
            None,
        )


class _PrefetchedEmbeddingFunction(
    torch.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        token_ids: torch.Tensor,
        local_weight: torch.Tensor,
        full_weight: torch.Tensor,
        full_shape: tuple[int, ...],
        full_numel: int,
        shard_numel: int,
        rank: int,
        world_size: int,
        compute_dtype: torch.dtype | None,
    ) -> torch.Tensor:
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
    def backward(ctx, grad_output: torch.Tensor):
        token_ids, local_weight = ctx.saved_tensors

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

            (full_grad_weight,) = torch.autograd.grad(
                outputs=dummy_output,
                inputs=dummy_weight,
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
            )

        local_grad_weight = _reduce_scatter_gradient(
            full_gradient=full_grad_weight,
            local_weight=local_weight,
            shard_numel=ctx.shard_numel,
            rank=ctx.rank,
            world_size=ctx.world_size,
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
            None,
        )


class PrefetchFSDPLinear(
    _PrefetchStateMixin,
    FSDPLinear,
):
    def __init__(self, original: FSDPLinear) -> None:
        nn.Module.__init__(self)

        self.weight = original.weight
        self._fsdp_full_shape = (
            original._fsdp_full_shape
        )
        self._fsdp_full_numel = (
            original._fsdp_full_numel
        )
        self._fsdp_shard_numel = (
            original._fsdp_shard_numel
        )
        self._fsdp_rank = original._fsdp_rank
        self._fsdp_world_size = (
            original._fsdp_world_size
        )
        self._fsdp_compute_dtype = (
            original._fsdp_compute_dtype
        )

        self._initialize_prefetch_state()
        self.train(original.training)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        full_weight = self.consume_full_weight()

        return _PrefetchedLinearFunction.apply(
            inputs,
            self.weight,
            full_weight,
            self._fsdp_full_shape,
            self._fsdp_full_numel,
            self._fsdp_shard_numel,
            self._fsdp_rank,
            self._fsdp_world_size,
            self._fsdp_compute_dtype,
        )


class PrefetchFSDPEmbedding(
    _PrefetchStateMixin,
    FSDPEmbedding,
):
    def __init__(self, original: FSDPEmbedding) -> None:
        nn.Module.__init__(self)

        self.weight = original.weight
        self._fsdp_full_shape = (
            original._fsdp_full_shape
        )
        self._fsdp_full_numel = (
            original._fsdp_full_numel
        )
        self._fsdp_shard_numel = (
            original._fsdp_shard_numel
        )
        self._fsdp_rank = original._fsdp_rank
        self._fsdp_world_size = (
            original._fsdp_world_size
        )
        self._fsdp_compute_dtype = (
            original._fsdp_compute_dtype
        )

        self._initialize_prefetch_state()
        self.train(original.training)

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        full_weight = self.consume_full_weight()

        return _PrefetchedEmbeddingFunction.apply(
            token_ids,
            self.weight,
            full_weight,
            self._fsdp_full_shape,
            self._fsdp_full_numel,
            self._fsdp_shard_numel,
            self._fsdp_rank,
            self._fsdp_world_size,
            self._fsdp_compute_dtype,
        )


class PrefetchFullyShardedDataParallel(
    FullyShardedDataParallel
):
    """
    FSDP with a forward all-gather lookahead of two sharded
    Linear/Embedding modules.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        # Build the already-tested synchronous FSDP representation.
        super().__init__(
            module=module,
            compute_dtype=compute_dtype,
        )

        # Replace the synchronous sharded modules while retaining
        # the exact same Parameter objects.
        self._replace_with_prefetch_layers(
            self.module
        )

        self._prefetch_layers = [
            submodule
            for submodule in self.module.modules()
            if isinstance(
                submodule,
                (
                    PrefetchFSDPLinear,
                    PrefetchFSDPEmbedding,
                ),
            )
        ]

        self._prefetch_hook_handles = []

        for index, layer in enumerate(
            self._prefetch_layers
        ):
            handle = layer.register_forward_hook(
                self._make_forward_post_hook(
                    index
                )
            )
            self._prefetch_hook_handles.append(
                handle
            )

    def _replace_with_prefetch_layers(
        self,
        parent: nn.Module,
    ) -> None:
        for name, child in list(
            parent.named_children()
        ):
            if isinstance(
                child,
                (
                    PrefetchFSDPLinear,
                    PrefetchFSDPEmbedding,
                ),
            ):
                continue

            if isinstance(child, FSDPLinear):
                setattr(
                    parent,
                    name,
                    PrefetchFSDPLinear(child),
                )

            elif isinstance(
                child,
                FSDPEmbedding,
            ):
                setattr(
                    parent,
                    name,
                    PrefetchFSDPEmbedding(child),
                )

            else:
                self._replace_with_prefetch_layers(
                    child
                )

    def _make_forward_post_hook(
        self,
        index: int,
    ):
        def hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            del module, inputs, output

            target_index = index + 2

            if target_index < len(
                self._prefetch_layers
            ):
                self._prefetch_layers[
                    target_index
                ].start_weight_prefetch()

        return hook

    def _clear_prefetch_state(self) -> None:
        for layer in self._prefetch_layers:
            layer.discard_unconsumed_prefetch()

    def forward(
        self,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        # Clear state left by an interrupted/dynamic forward.
        self._clear_prefetch_state()

        # The first two layers have no layers two positions
        # earlier, so start them at the beginning.
        for layer in self._prefetch_layers[:2]:
            layer.start_weight_prefetch()

        try:
            return super().forward(
                *inputs,
                **kwargs,
            )
        finally:
            # Normally every prefetch was consumed. This only waits
            # for unconsumed work under dynamic control flow/errors.
            self._clear_prefetch_state()

    @torch.no_grad()
    def finish_gradient_synchronization(
        self,
    ) -> None:
        self._clear_prefetch_state()
        super().finish_gradient_synchronization()
