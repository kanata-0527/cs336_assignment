import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_systems.naive_ddp import NaiveDDP


class FlatDDP(NaiveDDP):
    @torch.no_grad()
    def sync_gradients(self) -> None:
        grads = [
            parameter.grad
            for parameter in self.module.parameters()
            if parameter.grad is not None
        ]
        if not grads:
            return

        flat_grad = _flatten_dense_tensors(grads)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.div_(self.world_size)

        averaged_grads = _unflatten_dense_tensors(flat_grad, grads)
        for grad, averaged in zip(grads, averaged_grads, strict=True):
            grad.copy_(averaged)

class OverlapDDP(NaiveDDP):
    def __init__(self, module):
        super().__init__(module)
        self.pending = []

        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(
                    self.start_gradient_sync
                )

    def start_gradient_sync(self, parameter):
        grad = parameter.grad
        work = dist.all_reduce(
            grad, op=dist.ReduceOp.SUM, async_op=True
        )
        self.pending.append((grad, work))

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        for grad, work in self.pending:
            work.wait()
            grad.div_(self.world_size)

        self.pending.clear()

    def sync_gradients(self) -> None:
        # 兼容之前 benchmark 调用的接口。
        self.finish_gradient_synchronization()