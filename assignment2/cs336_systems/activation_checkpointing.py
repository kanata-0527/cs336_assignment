import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM


class CheckpointedTransformerLM(BasicsTransformerLM):
    def __init__(self, *, checkpoint_block_size: int = 0, **model_kwargs):
        if checkpoint_block_size < 0:
            raise ValueError("checkpoint_block_size must be non-negative")

        super().__init__(**model_kwargs)
        self.checkpoint_block_size = checkpoint_block_size

    def _forward_group(self, x, start: int, stop: int):
        """按顺序执行第 start 层到第 stop - 1 层。"""
        for index in range(start, stop):
            x = self.layers[index](x)
        return x

    def forward(self, token_ids):
        group_size = self.checkpoint_block_size
        if group_size == 0 or not torch.is_grad_enabled():
            return super().forward(token_ids)

        x = self.token_embeddings(token_ids)

        for start in range(0, len(self.layers), group_size):
            stop = min(start + group_size, len(self.layers))
            x = checkpoint(
                self._forward_group,
                x,
                start,
                stop,
                use_reentrant=False,
            )

        x = self.ln_final(x)
        return self.lm_head(x)