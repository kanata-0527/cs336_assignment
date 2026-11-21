import argparse
from itertools import product
from statistics import mean
from timeit import default_timer

import torch

from cs336_basics.model import scaled_dot_product_attention


def measure_attention(length, dim, warmup, steps, device, attention_fn):
    shape = (8, length, dim)
    inputs = tuple(
        torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
        for _ in range(3)
    )
    grad_output = torch.randn(shape, dtype=torch.float32, device=device)

    forward_times = []
    backward_times = []
    before_backward_mib = 0.0

    for step in range(warmup + steps):
        # 清空旧梯度，让每次反向都从相同的状态开始。
        for tensor in inputs:
            tensor.grad = None
        torch.cuda.synchronize(device)

        start = default_timer()
        output = attention_fn(*inputs)
        torch.cuda.synchronize(device)
        forward_ms = (default_timer() - start) * 1000

        # 只在第一轮正式测量中，读取前向结束时的显存。
        if step == warmup:
            before_backward_mib = torch.cuda.memory_allocated(device) / 1024**2

        start = default_timer()
        output.backward(grad_output)
        torch.cuda.synchronize(device)
        backward_ms = (default_timer() - start) * 1000

        if step >= warmup:
            forward_times.append(forward_ms)
            backward_times.append(backward_ms)

        del output

    return mean(forward_times), mean(backward_times), before_backward_mib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup 必须 >= 0，--steps 必须 >= 1")
    if not torch.cuda.is_available():
        parser.error("当前 PyTorch 环境无法使用 CUDA")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    print("T,d,forward_ms,backward_ms,before_backward_mib,status", flush=True)

    for length, dim in product(args.lengths, args.dims):
        attention_fn = scaled_dot_product_attention
        if args.compile:
            # Each shape gets its own compiled graph, even in a single-process sweep.
            torch.compiler.reset()
            attention_fn = torch.compile(attention_fn, dynamic=False)

        try:
            forward_ms, backward_ms, memory_mib = measure_attention(
                length, dim, args.warmup, args.steps, device, attention_fn
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{length},{dim},,,,OOM", flush=True)
        else:
            print(
                f"{length},{dim},{forward_ms:.3f},{backward_ms:.3f},"
                f"{memory_mib:.2f},ok",
                flush=True,
            )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
