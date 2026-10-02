from __future__ import annotations

from typing import Any, Iterable, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    Optimizer-state sharding.

    Every rank retains a full copy of the model parameters and gradients,
    but optimizer state is created only for the parameters assigned to
    the current rank.

    After each local optimizer step, updated parameters are broadcast
    from their owner ranks so every model replica remains identical.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        if not dist.is_available():
            raise RuntimeError(
                "torch.distributed is not available"
            )

        if not dist.is_initialized():
            raise RuntimeError(
                "Initialize the distributed process group before "
                "constructing ShardedOptimizer"
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)

        # Must exist before Optimizer.__init__, because the superclass
        # invokes our overridden add_param_group().
        self._local_optimizer: Optimizer | None = None
        self._pending_local_param_groups: list[
            dict[str, Any]
        ] = []

        self._ordered_parameters: list[torch.Tensor] = []
        self._parameter_owners: dict[int, int] = {}

        # Track assigned parameter elements to balance optimizer state
        # by memory rather than merely by number of tensors.
        self._rank_numels = [
            0 for _ in range(self.world_size)
        ]

        # Register the complete parameter set with this wrapper.
        # Optimizer.__init__ calls self.add_param_group().
        super().__init__(params, defaults={})

        if self._pending_local_param_groups:
            self._local_optimizer = optimizer_cls(
                self._pending_local_param_groups,
                **self.optimizer_kwargs,
            )

            # Expose local optimizer state through the wrapper.
            self.state = self._local_optimizer.state
            self.defaults = self._local_optimizer.defaults

        self._pending_local_param_groups.clear()

    def _assign_owner(
        self,
        parameter: torch.Tensor,
    ) -> int:
        parameter_id = id(parameter)

        existing_owner = self._parameter_owners.get(
            parameter_id
        )
        if existing_owner is not None:
            return existing_owner

        # Greedy load balancing by parameter size.
        owner = min(
            range(self.world_size),
            key=lambda rank: (
                self._rank_numels[rank],
                rank,
            ),
        )

        self._parameter_owners[parameter_id] = owner
        self._rank_numels[owner] += parameter.numel()
        self._ordered_parameters.append(parameter)

        return owner

    def add_param_group(
        self,
        param_group: dict[str, Any],
    ) -> None:
        """
        Register a complete parameter group with the wrapper and add
        only the current rank's shard to the wrapped optimizer.
        """
        group = dict(param_group)

        raw_params = group["params"]

        if isinstance(raw_params, torch.Tensor):
            parameters = [raw_params]
        else:
            parameters = list(raw_params)

        group["params"] = parameters

        # Register all parameters in the public wrapper. This also
        # performs PyTorch's duplicate-parameter validation.
        super().add_param_group(group)

        registered_group = self.param_groups[-1]

        local_parameters: list[torch.Tensor] = []

        for parameter in parameters:
            owner = self._assign_owner(parameter)

            if owner == self.rank:
                local_parameters.append(parameter)

        if not local_parameters:
            return

        local_group = {
            key: value
            for key, value in registered_group.items()
            if key != "params"
        }
        local_group["params"] = local_parameters

        if self._local_optimizer is None:
            # During Optimizer.__init__, defer construction until all
            # initial parameter groups have been processed.
            self._pending_local_param_groups.append(
                local_group
            )
        else:
            # Support parameter groups added during training.
            self._local_optimizer.add_param_group(
                local_group
            )
            self.state = self._local_optimizer.state

    def step(
        self,
        closure=None,
        **kwargs: Any,
    ):
        """
        Update the local parameter shard and synchronize all updated
        parameters across ranks.
        """
        loss = None

        if self._local_optimizer is not None:
            if closure is None:
                loss = self._local_optimizer.step(
                    **kwargs
                )
            else:
                loss = self._local_optimizer.step(
                    closure=closure,
                    **kwargs,
                )

        # All ranks invoke broadcasts in exactly the same order.
        with torch.no_grad():
            for parameter in self._ordered_parameters:
                owner = self._parameter_owners[
                    id(parameter)
                ]

                dist.broadcast(
                    parameter.data,
                    src=owner,
                )

        return loss

    def state_dict(self) -> dict[str, Any]:
        """
        Return only this rank's optimizer-state shard.
        """
        if self._local_optimizer is None:
            return {
                "state": {},
                "param_groups": [],
            }

        return self._local_optimizer.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        if self._local_optimizer is None:
            if state_dict.get("state"):
                raise RuntimeError(
                    "Cannot load non-empty optimizer state on "
                    "a rank with no assigned parameters"
                )
            return

        self._local_optimizer.load_state_dict(
            state_dict
        )
        self.state = self._local_optimizer.state
