import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    num_starts = len(dataset) - context_length

    starts = np.random.randint(0, num_starts, size=batch_size) # 随机抽一个起点
    inputs = np.empty((batch_size, context_length), dtype=np.int64)
    targets = np.empty_like(inputs)

    for row, start in enumerate(starts): # 样本写进
        end = start + context_length
        inputs[row] = dataset[start:end]
        targets[row] = dataset[start + 1: end + 1]

    x = torch.as_tensor(inputs, dtype=torch.long, device=device) # x 和 y 转成张量
    y = torch.as_tensor(targets, dtype=torch.long, device=device)
    return x,y  