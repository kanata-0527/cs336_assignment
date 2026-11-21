"""FlashAttention-2 forward with one Triton program per query tile and batch item."""

import math

import torch
import triton
import triton.language as tl

from cs336_systems.flash_attention import validate_attention_inputs


@triton.jit
def _flash_forward(
    q_ptr, k_ptr, v_ptr, output_ptr, lse_ptr,
    q_batch_stride, q_row_stride, q_dim_stride,
    k_batch_stride, k_row_stride, k_dim_stride,
    v_batch_stride, v_row_stride, v_dim_stride,
    o_batch_stride, o_row_stride, o_dim_stride,
    lse_batch_stride, lse_row_stride,
    n_queries, n_keys, scale,
    HEAD_DIM: tl.constexpr,
    D_TILE: tl.constexpr,
    Q_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_start = tl.program_id(0) * Q_TILE
    batch = tl.program_id(1)

    q_block = tl.make_block_ptr(
        q_ptr + batch * q_batch_stride, (n_queries, HEAD_DIM),
        (q_row_stride, q_dim_stride), (query_start, 0), (Q_TILE, D_TILE), (1, 0),
    )
    k_block = tl.make_block_ptr(
        k_ptr + batch * k_batch_stride, (n_keys, HEAD_DIM),
        (k_row_stride, k_dim_stride), (0, 0), (K_TILE, D_TILE), (1, 0),
    )
    v_block = tl.make_block_ptr(
        v_ptr + batch * v_batch_stride, (n_keys, HEAD_DIM),
        (v_row_stride, v_dim_stride), (0, 0), (K_TILE, D_TILE), (1, 0),
    )
    output_block = tl.make_block_ptr(
        output_ptr + batch * o_batch_stride, (n_queries, HEAD_DIM),
        (o_row_stride, o_dim_stride), (query_start, 0), (Q_TILE, D_TILE), (1, 0),
    )
    lse_block = tl.make_block_ptr(
        lse_ptr + batch * lse_batch_stride, (n_queries,),
        (lse_row_stride,), (query_start,), (Q_TILE,), (0,),
    )

    query = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")
    query_positions = query_start + tl.arange(0, Q_TILE)
    row_max = tl.full((Q_TILE,), -float("inf"), tl.float32)
    normalizer = tl.zeros((Q_TILE,), tl.float32)
    output_acc = tl.zeros((Q_TILE, D_TILE), tl.float32)

    key_limit = n_keys
    if is_causal:
        key_limit = tl.minimum(n_keys, query_start + Q_TILE)

    for key_tile_index in range(tl.cdiv(key_limit, K_TILE)):
        key = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
        value = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
        key_positions = key_tile_index * K_TILE + tl.arange(0, K_TILE)
        scores = tl.dot(query, tl.trans(key), input_precision="tf32x3") * scale
        if is_causal:
            scores += tl.where(query_positions[:, None] >= key_positions[None, :], 0.0, -1.0e6)
        scores = tl.where(key_positions[None, :] < n_keys, scores, -float("inf"))

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        previous_scale = tl.exp(row_max - new_max)
        unnormalized = tl.exp(scores - new_max[:, None])
        normalizer = previous_scale * normalizer + tl.sum(unnormalized, axis=1)
        output_acc = tl.dot(
            unnormalized.to(value.dtype), value,
            acc=previous_scale[:, None] * output_acc, input_precision="tf32x3",
        )
        row_max = new_max

        k_block = tl.advance(k_block, (K_TILE, 0))
        v_block = tl.advance(v_block, (K_TILE, 0))

    output = output_acc / normalizer[:, None]
    logsumexp = row_max + tl.log(normalizer)
    tl.store(output_block, output.to(output_block.type.element_ty), boundary_check=(0, 1))
    tl.store(lse_block, logsumexp, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        validate_attention_inputs(q, k, v)
        if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Triton attention requires CUDA inputs with dtype float16, bfloat16 or float32")

        batch_size = math.prod(q.shape[:-2])
        n_queries, head_dim = q.shape[-2:]
        n_keys = k.shape[-2]
        q_flat = q.reshape(batch_size, n_queries, head_dim)
        k_flat = k.reshape(batch_size, n_keys, head_dim)
        v_flat = v.reshape(batch_size, n_keys, head_dim)
        output = torch.empty((batch_size, n_queries, head_dim), dtype=q.dtype, device=q.device)
        logsumexp = torch.empty((batch_size, n_queries), dtype=torch.float32, device=q.device)

        q_tile, k_tile = 32, 64
        if output.numel() != 0:
            _flash_forward[(triton.cdiv(n_queries, q_tile), batch_size)](
                q_flat, k_flat, v_flat, output, logsumexp,
                *q_flat.stride(), *k_flat.stride(), *v_flat.stride(),
                *output.stride(), *logsumexp.stride(),
                n_queries, n_keys, 1 / math.sqrt(head_dim),
                HEAD_DIM=head_dim, D_TILE=max(16, triton.next_power_of_2(head_dim)),
                Q_TILE=q_tile, K_TILE=k_tile, is_causal=is_causal,
            )

        output = output.reshape(q.shape)
        logsumexp = logsumexp.reshape(q.shape[:-1])
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward is implemented in the next assignment part")
