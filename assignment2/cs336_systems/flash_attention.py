"""Tiled PyTorch reference for FlashAttention-2."""

import math

import torch


_QUERY_TILE_SIZE = 32
_KEY_TILE_SIZE = 64


def validate_attention_inputs(q, k, v):
    if (
        q.ndim < 2
        or k.ndim != q.ndim
        or k.shape != v.shape
        or q.shape[:-2] != k.shape[:-2]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("Q, K and V must have matching batch and head dimensions; K and V must have matching shapes")
    if q.shape[-1] == 0 or k.shape[-2] == 0:
        raise ValueError("Attention requires a positive head dimension and at least one key")
    if not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K and V must share a floating-point dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K and V must be on the same device")


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        validate_attention_inputs(q, k, v)
        n_queries, head_dim = q.shape[-2:]
        n_keys = k.shape[-2]
        scale = 1 / math.sqrt(head_dim)
        accum_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        output = torch.empty_like(q)
        logsumexp = torch.empty(q.shape[:-1], dtype=accum_dtype, device=q.device)

        for query_start in range(0, n_queries, _QUERY_TILE_SIZE):
            query_end = min(query_start + _QUERY_TILE_SIZE, n_queries)
            query_tile = q[..., query_start:query_end, :].to(accum_dtype)
            row_max = query_tile.new_full(query_tile.shape[:-1], -torch.inf)
            normalizer = torch.zeros_like(row_max)
            output_acc = torch.zeros_like(query_tile)
            key_end = n_keys

            if is_causal:
                query_positions = torch.arange(query_start, query_end, device=q.device)
                key_end = min(n_keys, query_end)

            for key_start in range(0, key_end, _KEY_TILE_SIZE):
                key_stop = min(key_start + _KEY_TILE_SIZE, key_end)
                key_tile = k[..., key_start:key_stop, :].to(accum_dtype)
                value_tile = v[..., key_start:key_stop, :].to(accum_dtype)
                scores = (query_tile @ key_tile.transpose(-2, -1)) * scale

                if is_causal:
                    key_positions = torch.arange(key_start, key_stop, device=q.device)
                    future = key_positions[None, :] > query_positions[:, None]
                    scores = scores.masked_fill(future, -torch.inf)

                # Rescale the previous sums whenever the running row maximum changes.
                new_max = torch.maximum(row_max, scores.amax(dim=-1))
                previous_scale = (row_max - new_max).exp()
                unnormalized = (scores - new_max[..., None]).exp()
                normalizer = previous_scale * normalizer + unnormalized.sum(dim=-1)
                output_acc = previous_scale[..., None] * output_acc + unnormalized @ value_tile
                row_max = new_max

            output[..., query_start:query_end, :] = output_acc / normalizer[..., None]
            logsumexp[..., query_start:query_end] = row_max + normalizer.log()

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward is implemented in the next assignment part")
