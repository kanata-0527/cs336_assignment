import torch

def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    max_logit = logits.amax(dim=-1, keepdim=True)
    shifted = logits - max_logit

    # 给每个目标 ID 加一个长度为 1 的词表轴，方便 gather 取值。
    indices = targets.to(device=logits.device, dtype=torch.long).unsqueeze(-1)
    target_scores = shifted.gather(dim=-1, index=indices).squeeze(-1)

    log_denominator = shifted.exp().sum(dim=-1).log()
    losses = log_denominator - target_scores
    return losses.mean()