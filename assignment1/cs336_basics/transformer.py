import torch
from torch import nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSnorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.swiglu import SwiGLU, silu

class SiLUFeedForward(nn.Module): # 加入 transformer Block
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))
        

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None,
        device=None,
        dtype=None,
        *,
        norm_style: str = "pre",
        ffn_style: str = "swiglu",
    ):
        super().__init__()
        self.norm_style = norm_style

        if norm_style != "none":
            self.ln1 = RMSnorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSnorm(d_model, device=device, dtype=dtype)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )

        ffn_class = SwiGLU if ffn_style == "swiglu" else SiLUFeedForward
        self.ffn = ffn_class(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_style == "pre":
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))

        if self.norm_style == "post":
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))

        x = x + self.attn(x)
        return x + self.ffn(x)

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
        *,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_style: str = "swiglu",
        tie_embeddings: bool = False,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype
        )

        rope = None
        if use_rope:
            rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=d_model // num_heads,
                max_seq_len=context_length,
                device=device,
            )
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=rope,
                device=device,
                dtype=dtype,
            )
            self.layers.append(block)

        self.ln_final = RMSnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)