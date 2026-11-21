import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

import math

from cs336_basics.softmax import softmax

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None=None
)-> torch.Tensor:
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is None:
        weights = softmax(scores, dim=-1)
    else:
        scores = scores.masked_fill(~mask, float("-inf"))

        # 整行都被遮住时，先给 softmax 一行有限的分数，再清零权重。
        fully_masked = ~mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)
        weights = softmax(scores, dim=-1).masked_fill(~mask, 0.0)

    return weights @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., T, d_model) -> (..., T, H, head_dim) -> (..., H, T, head_dim)
        x = x.unflatten(-1, (self.num_heads, self.head_dim))
        return x.transpose(-3, -2)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        indices = torch.arange(x.shape[-2], device=x.device)
        if self.rope is not None:
            if token_positions is None:
                token_positions = indices

            # (..., T) -> (..., 1, T)，同一组位置广播给所有头。
            positions = token_positions.to(device=x.device, dtype=torch.long)
            positions = positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        causal_mask = indices[:, None] >= indices[None, :]
        heads = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # (..., H, T, head_dim) -> (..., T, H, head_dim) -> (..., T, d_model)
        merged = heads.transpose(-3, -2).flatten(start_dim=-2)
        return self.output_proj(merged)