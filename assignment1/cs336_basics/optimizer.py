import math
import torch
from collections.abc import Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        beta1, beta2 = betas
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                # 权重衰减使用基础学习率 lr。
                parameter.mul_(1.0 - lr * weight_decay)

                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * grad**2
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                step_size = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                denominator = v.sqrt() + eps
                parameter.sub_(step_size * m / denominator)

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)

    if it >= cosine_cycle_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_weight * (max_learning_rate - min_learning_rate)

@torch.no_grad()
def gradient_clipping( # 梯度裁剪
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return

    norm_device = grads[0].device
    squared_norms = []

    for grad in grads:
        grad_for_norm = grad
        if grad.dtype in (torch.float16, torch.bfloat16):
            grad_for_norm = grad.float()

        squared_norms.append(grad_for_norm.square().sum().to(norm_device))

    total_norm = torch.stack(squared_norms).sum().sqrt()
    if total_norm < max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for grad in grads:
        grad.mul_(scale.to(grad.device))