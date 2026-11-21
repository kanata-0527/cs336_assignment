import math

import torch

from cs336_basics.softmax import softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


END_OF_TEXT = "<|endoftext|>"


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> int:
    """根据一个位置的词表分数，抽出一个 token ID。"""
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # 减最大值保持相对分数，再用温度调整概率的集中程度。
    scaled_logits = (logits - logits.max()) / temperature
    probs = softmax(scaled_logits, dim=-1)

    if top_p == 1.0:
        return int(torch.multinomial(probs, num_samples=1).item())

    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = sorted_probs.cumsum(dim=0)

    # 看的是加上当前 token 之前的累计概率。
    # 这样让累计概率首次达到 top_p 的那个 token 也会被保留。
    cumulative_before = torch.cat((cumulative.new_zeros((1,)), cumulative[:-1]))
    kept_probs = sorted_probs.masked_fill(cumulative_before >= top_p, 0.0)
    kept_probs = kept_probs / kept_probs.sum()

    sampled_index = torch.multinomial(kept_probs, num_samples=1)
    return int(sorted_ids[sampled_index].item())


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """返回开头和模型续写的文本。"""
    token_ids = tokenizer.encode(prompt)

    end_ids = tokenizer.encode(END_OF_TEXT)
    end_id = end_ids[0]

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            # 超过模型上下文长度时，使用最近的 token 继续预测。
            context = token_ids[-model.context_length :]
            inputs = torch.tensor([context], dtype=torch.long, device=device)

            # logits 形状为 (1, 序列长度, 词表大小)。
            # [0, -1] 取唯一一条样本的最后一个位置，保留整张词表的分数。
            logits = model(inputs)
            next_token = _sample_next_token(logits[0, -1], temperature, top_p)

            if next_token == end_id:
                break

            token_ids.append(next_token)
    finally:
        model.train(was_training)

    return tokenizer.decode(token_ids)