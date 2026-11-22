from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Shard per-parameter optimizer state across replicas in the default process group.

    Every rank supplies parameters in the same order and synchronizes gradients
    before calling step(). Optimizer checkpoints are local to each rank.
    """

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._owners: dict[torch.Tensor, int] = {}
        self._rank_sizes = [0] * self.world_size
        self._local_optimizer: Optimizer | None = None

        # Optimizer.__init__ calls our add_param_group with each global group.
        super().__init__(params, {})
        local_groups = [
            self._local_group(group, self._owners) for group in self.param_groups
        ]
        self._local_optimizer = optimizer_cls(local_groups, **kwargs)
        self.defaults = self._local_optimizer.defaults
        self._copy_local_options()

    @staticmethod
    def _options(group: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in group.items()
            if name not in ("params", "param_names")
        }

    def _local_group(
        self, group: dict[str, Any], owners: dict[torch.Tensor, int]
    ) -> dict[str, Any]:
        local_group = self._options(group)
        local_group["params"] = [
            parameter for parameter in group["params"] if owners[parameter] == self.rank
        ]
        if "param_names" in group:
            local_group["param_names"] = [
                name
                for parameter, name in zip(group["params"], group["param_names"], strict=True)
                if owners[parameter] == self.rank
            ]
        return local_group

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group.copy())
        group = self.param_groups[-1]

        new_owners = {}
        rank_sizes = self._rank_sizes.copy()
        try:
            for parameter in group["params"]:
                if parameter not in new_owners:
                    owner = min(range(self.world_size), key=rank_sizes.__getitem__)
                    new_owners[parameter] = owner
                    rank_sizes[owner] += parameter.numel()

            if self._local_optimizer is not None:
                self._local_optimizer.add_param_group(self._local_group(group, new_owners))
                group.update(self._options(self._local_optimizer.param_groups[-1]))
        except Exception:
            self.param_groups.pop()
            raise

        self._owners.update(new_owners)
        self._rank_sizes = rank_sizes

    def _copy_local_options(self) -> None:
        # Expose optimizer state and defaults while keeping all model parameters
        # in the wrapper's groups so Optimizer.zero_grad clears every gradient.
        self.state = self._local_optimizer.state
        for group, local_group in zip(
            self.param_groups, self._local_optimizer.param_groups, strict=True
        ):
            group.update(self._options(local_group))

    def _apply_group_options(self) -> None:
        # Learning-rate schedulers update the wrapper's parameter groups.
        for group, local_group in zip(
            self.param_groups, self._local_optimizer.param_groups, strict=True
        ):
            local_group.update(self._options(group))

    def step(self, closure=None, **kwargs: Any):
        self._apply_group_options()
        loss = self._local_optimizer.step(closure=closure, **kwargs)

        if self.world_size > 1:
            with torch.no_grad():
                for parameter, owner in self._owners.items():
                    dist.broadcast(parameter, src=owner)

        return loss

    def state_dict(self) -> dict[str, Any]:
        """Return this rank's optimizer checkpoint using the local optimizer format."""
        self._apply_group_options()
        return self._local_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the checkpoint saved for this rank and parameter assignment."""
        self._local_optimizer.load_state_dict(state_dict)
        self._copy_local_options()
