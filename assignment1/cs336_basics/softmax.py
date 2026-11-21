import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    input_dtype = x.dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    max_value = x.amax(dim=dim, keepdim=True)
    shifted = (x - max_value).exp()
    sum_exp = shifted.sum(dim=dim, keepdim=True)
    probabilities = shifted / sum_exp

    return probabilities.to(input_dtype)