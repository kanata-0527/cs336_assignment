import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")

        pair_dims = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_freq = theta ** (-pair_dims / d_k)

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_freq[None, :]

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 把相邻的两维配成一对
        even = x[..., 0::2]
        odd = x[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # 先把每对放在一起，再展平回原来的最后一维。
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rotated.flatten(start_dim=-2).to(dtype=x.dtype)