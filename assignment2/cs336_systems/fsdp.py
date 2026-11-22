from contextlib import nullcontext
from dataclasses import dataclass
from inspect import signature

import torch
import torch.distributed as dist
from torch import nn

from cs336_basics.model import Embedding, Linear


@dataclass
class _Gather:
    full: torch.Tensor
    source: torch.Tensor
    work: dist.Work | None

    def finish(self) -> torch.Tensor:
        if self.work is not None:
            self.work.wait()
        return self.full


@dataclass
class _Gradient:
    shard: nn.Parameter
    value: torch.Tensor
    source: torch.Tensor
    work: dist.Work


class _ShardedLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shard, bias, inputs, fsdp, layer, run):
        ctx.save_for_backward(shard, bias, inputs)
        ctx.fsdp = fsdp
        ctx.layer = layer
        ctx.run = run
        ctx.autocast = torch.is_autocast_enabled(shard.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(shard.device.type)
        weight = fsdp._forward_weight(layer)
        # Autograd saves only the shard and input. The full weight dies here.
        with fsdp._mark("fsdp_forward_compute", shard, layer):
            return run(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        shard, bias, inputs = ctx.saved_tensors
        fsdp = ctx.fsdp
        with fsdp._mark("fsdp_backward_all_gather", shard, ctx.layer):
            weight = fsdp._gather(shard, fsdp.compute_dtype).finish()
            weight = weight[: fsdp._shapes[shard][0]]

        need_weight, need_bias, need_input = ctx.needs_input_grad[:3]
        weight = weight.detach().requires_grad_(need_weight)
        bias = bias.detach().requires_grad_(need_bias) if bias is not None else None
        inputs = inputs.detach().requires_grad_(need_input)
        leaves = (weight, bias, inputs)
        requested = [tensor for tensor in leaves if tensor is not None and tensor.requires_grad]

        # Rebuild the local layer's backward graph from its input and gathered
        # weight. Gradients of the surrounding model still flow normally.
        with torch.enable_grad(), torch.autocast(
            shard.device.type, dtype=ctx.autocast_dtype, enabled=ctx.autocast
        ):
            out = ctx.run(inputs, weight, bias)
        gradients = iter(torch.autograd.grad(out, requested, grad_output))
        weight_grad, bias_grad, input_grad = (
            next(gradients) if tensor is not None and tensor.requires_grad else None
            for tensor in leaves
        )

        if weight_grad is not None:
            fsdp._reduce_weight_gradient(shard, weight_grad)
        return None, bias_grad, input_grad, None, None, None


class FSDP(nn.Module):
    """Shard dense Linear/Embedding weights across the default process group.

    Pass device to move and shard one parameter at a time during setup.
    Ranks must run the same sequence of layers and finish gradient synchronization
    after loss.backward().
    One local layer is recomputed in backward to avoid keeping full weights.
    """

    def __init__(
        self, module: nn.Module, compute_dtype: torch.dtype | None = None,
        *, nvtx: bool = False, device: torch.device | None = None,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.nvtx = nvtx
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        self._shapes: dict[nn.Parameter, torch.Size] = {}
        self._previous_order: list[nn.Module] = []
        self._current_order: list[nn.Module] = []
        self._prefetch: tuple[nn.Module, _Gather] | None = None
        self._pending_grad: _Gradient | None = None
        self._forward_wait_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] | None = None
        self._layer_names = {
            layer: name or type(layer).__name__
            for name, layer in module.named_modules()
            if isinstance(layer, (Linear, Embedding, nn.Linear, nn.Embedding))
        }
        sharded = {layer.weight for layer in self._layer_names}
        replacements = {}
        with torch.no_grad():
            for parameter in module.parameters():
                dtype = torch.float32 if compute_dtype is not None and parameter.is_floating_point() else parameter.dtype
                full = parameter.detach().to(device=device, dtype=dtype)
                if self.world_size > 1:
                    dist.broadcast(full, src=0)

                shape = full.shape
                if parameter in sharded:
                    rows = (full.size(0) + self.world_size - 1) // self.world_size
                    start = rows * rank
                    local = full.new_zeros((rows, *shape[1:]))
                    shard_rows = min(rows, max(0, full.size(0) - start))
                    local[:shard_rows].copy_(full[start : start + shard_rows])
                else:
                    local = full

                if parameter.device == local.device:
                    parameter.data = local
                    master = parameter
                else:
                    master = nn.Parameter(local, requires_grad=parameter.requires_grad)
                    replacements[parameter] = master

                if parameter in sharded:
                    self._shapes[master] = shape
                del full, local

            if replacements:
                # Apply the same replacement to every alias of a tied weight.
                for layer in module.modules():
                    for name, parameter in layer.named_parameters(recurse=False, remove_duplicate=False):
                        if parameter in replacements:
                            layer._parameters[name] = replacements[parameter]
            if device is not None:
                module.to(device)
            for layer in self._layer_names:
                self._wrap_layer(layer)

    def _mark(self, name: str, tensor: torch.Tensor, layer: nn.Module | None = None):
        if self.nvtx and tensor.is_cuda:
            label = f"{name}:{self._layer_names[layer]}" if layer is not None else name
            return torch.cuda.nvtx.range(label)
        return nullcontext()

    def start_forward_wait_timing(self) -> None:
        """Measure the wait on the compute stream, including CUDA-event overhead."""
        self._forward_wait_events = []

    def finish_forward_wait_timing(self) -> list[tuple[str, float]]:
        events = self._forward_wait_events
        self._forward_wait_events = None
        if events is None:
            return []
        return [(layer, start.elapsed_time(end)) for layer, start, end in events]

    def _wrap_layer(self, layer: nn.Module) -> None:
        original_forward = layer.forward
        forward_signature = signature(original_forward)
        input_name = next(iter(forward_signature.parameters))
        has_bias = "bias" in layer._parameters

        def run(inputs, weight, bias):
            original_weight = layer.weight
            original_bias = layer.bias if has_bias else None
            layer._parameters["weight"] = weight
            try:
                if has_bias and bias is not None:
                    layer._parameters["bias"] = bias.to(weight.dtype)
                if inputs.is_floating_point() and self.compute_dtype is not None:
                    inputs = inputs.to(weight.dtype)
                return original_forward(inputs)
            finally:
                layer._parameters["weight"] = original_weight
                if has_bias:
                    layer._parameters["bias"] = original_bias

        def wrapped_forward(*args, **kwargs):
            if len(args) == 1 and not kwargs:
                inputs = args[0]
            else:
                inputs = forward_signature.bind(*args, **kwargs).arguments[input_name]
            return _ShardedLayer.apply(layer.weight, getattr(layer, "bias", None), inputs, self, layer, run)

        layer.forward = wrapped_forward

    def _gather(self, shard, dtype) -> _Gather:
        source = shard.detach().to(dtype=dtype or shard.dtype)
        if self.world_size == 1 or source.numel() == 0:
            return _Gather(source, source, None)

        full = source.new_empty((source.size(0) * self.world_size, *source.shape[1:]))
        work = dist.all_gather_into_tensor(full, source, async_op=True)
        return _Gather(full, source, work)

    def _clear_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch[1].finish()
            self._prefetch = None

    def _forward_weight(self, layer):
        shard = layer.weight
        if self._prefetch is not None and self._prefetch[0] is layer:
            _, gather = self._prefetch
            self._prefetch = None
        else:
            self._clear_prefetch()
            with self._mark("fsdp_forward_all_gather", shard, layer):
                gather = self._gather(shard, self.compute_dtype)

        index = len(self._current_order)
        self._current_order.append(layer)
        # Once the preceding layer has completed, launch at most one gather
        # ahead. Learn the execution order from the previous full forward pass.
        if index + 1 < len(self._previous_order) and self._previous_order[index] is layer:
            following = self._previous_order[index + 1]
            with self._mark("fsdp_forward_prefetch", following.weight, following):
                self._prefetch = (following, self._gather(following.weight, self.compute_dtype))

        events = self._forward_wait_events
        if events is not None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        with self._mark("fsdp_forward_wait", shard, layer):
            full = gather.finish()
        if events is not None:
            end_event.record()
            events.append((self._layer_names[layer], start_event, end_event))
        return full[: self._shapes[shard][0]]

    def forward(self, *inputs, **kwargs):
        self._current_order = []
        try:
            output = self.module(*inputs, **kwargs)
        finally:
            self._clear_prefetch()
        self._previous_order = self._current_order
        return output

    @torch.no_grad()
    def _store_shard_gradient(self, shard, value) -> None:
        value.div_(self.world_size)
        if shard.grad is None:
            shard.grad = value
        else:
            shard.grad.add_(value)

    def _finish_pending_gradient(self) -> None:
        if self._pending_grad is not None:
            pending = self._pending_grad
            pending.work.wait()
            self._store_shard_gradient(pending.shard, pending.value)
            self._pending_grad = None

    def _reduce_weight_gradient(self, shard, gradient) -> None:
        gradient = gradient.to(shard.dtype).contiguous()
        if self.world_size == 1 or shard.numel() == 0:
            self._store_shard_gradient(shard, gradient)
            return

        # Retire the previous collective before keeping another full gradient
        # buffer alive. This limits the memory used by unfinished collectives.
        self._finish_pending_gradient()
        padded_rows = shard.size(0) * self.world_size
        if gradient.size(0) < padded_rows:
            padded = gradient.new_zeros((padded_rows, *gradient.shape[1:]))
            padded[: gradient.size(0)].copy_(gradient)
            gradient = padded
        local = torch.empty_like(shard)
        with self._mark("fsdp_reduce_scatter", shard):
            work = dist.reduce_scatter_tensor(local, gradient, async_op=True)
        self._pending_grad = _Gradient(shard, local, gradient, work)

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        self._finish_pending_gradient()
        if self.world_size > 1:
            for parameter in self.module.parameters():
                if parameter not in self._shapes and parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(self.world_size)

    def sync_gradients(self) -> None:
        """Use the same training-loop entry point as the DDP benchmark."""
        self.finish_gradient_synchronization()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params = {}
        for name, parameter in self.module.named_parameters():
            if parameter in self._shapes:
                full = self._gather(parameter, None).finish()
                full_params[name] = full[: self._shapes[parameter][0]]
            else:
                full_params[name] = parameter.detach().clone()
        return full_params
