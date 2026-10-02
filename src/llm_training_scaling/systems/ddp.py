from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    """
    Minimal distributed data-parallel wrapper.

    Each rank stores a full model replica. Model state is initially
    broadcast from rank 0. After backward(), parameter gradients are
    individually all-reduced and averaged across ranks.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is not available"
            )

        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize the distributed process group "
                "before constructing NaiveDDP"
            )

        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._broadcast_model_state()

    @torch.no_grad()
    def _broadcast_model_state(self) -> None:
        """
        Make all model replicas identical to rank 0.

        Parameters and buffers are both broadcast. Parameters that do
        not require gradients still need identical initial values.
        """
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)

        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """
        Synchronize gradients after backward and before optimizer.step.
        """
        if self.world_size == 1:
            return

        for parameter in self.module.parameters():
            gradient = parameter.grad

            # Parameters unused in this graph, or parameters with
            # requires_grad=False, may not have a gradient.
            if gradient is None:
                continue

            dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                async_op=False,
            )
            gradient.div_(self.world_size)


class FlatGradientDDP(NaiveDDP):
    """
    Minimal DDP variant that flattens all available gradients and
    synchronizes them with one all-reduce operation.
    """

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        if self.world_size == 1:
            return

        gradients = [
            parameter.grad
            for parameter in self.module.parameters()
            if parameter.grad is not None
        ]

        if not gradients:
            return

        flat_gradient = torch._utils._flatten_dense_tensors(
            gradients
        )

        dist.all_reduce(
            flat_gradient,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )

        flat_gradient.div_(self.world_size)

        synchronized_gradients = (
            torch._utils._unflatten_dense_tensors(
                flat_gradient,
                gradients,
            )
        )

        for original, synchronized in zip(
            gradients,
            synchronized_gradients,
            strict=True,
        ):
            original.copy_(synchronized)


class OverlapDDP(NaiveDDP):
    """
    DDP that launches an asynchronous all-reduce as soon as each
    parameter gradient has finished accumulating.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__(module)

        self._pending_reductions: list[
            tuple[dist.Work, torch.Tensor]
        ] = []

        self._hook_handles: list[
            torch.utils.hooks.RemovableHandle
        ] = []

        for parameter in self.module.parameters():
            if not parameter.requires_grad:
                continue

            handle = parameter.register_post_accumulate_grad_hook(
                self._make_gradient_hook()
            )
            self._hook_handles.append(handle)

    def _make_gradient_hook(self):
        @torch.no_grad()
        def hook(parameter: torch.Tensor) -> None:
            gradient = parameter.grad

            if gradient is None:
                return

            work = dist.all_reduce(
                gradient,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )

            self._pending_reductions.append(
                (work, gradient)
            )

        return hook

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        """
        Wait for asynchronous reductions, then average gradients.
        """
        for work, gradient in self._pending_reductions:
            work.wait()
            gradient.div_(self.world_size)

        self._pending_reductions.clear()
