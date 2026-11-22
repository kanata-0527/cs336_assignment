import torch
import torch.distributed as dist
from torch import nn

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        # 所有进程都调用 broadcast，让 rank 0 的参数成为共同起点。
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @torch.no_grad()
    def sync_gradients(self):
        for parameter in self.module.parameters():
            grad = parameter.grad
            if grad is None:
                continue

            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)