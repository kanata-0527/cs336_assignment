import argparse
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def benchmark_size(
    size_mb: int, device: torch.device, warmup: int, steps: int
) -> torch.Tensor:
    # Each float32 element is 4 bytes. Zeros can be summed in-place repeatedly.
    numel = size_mb * 1_000_000 // 4
    data = torch.zeros(numel, dtype=torch.float32, device=device)

    for _ in range(warmup):
        dist.all_reduce(data)
    torch.cuda.synchronize(device)

    elapsed_ms = []
    for _ in range(steps):
        # Line up ranks and finish the barrier before starting the timer.
        dist.barrier()
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        dist.all_reduce(data)
        torch.cuda.synchronize(device)
        elapsed_ms.append((time.perf_counter() - start) * 1_000)

    # Reduce the per-step timings separately from the measured tensor.
    timings = torch.tensor(elapsed_ms, dtype=torch.float64, device=device)
    dist.reduce(timings, dst=0, op=dist.ReduceOp.MAX)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark single-node NCCL all-reduce.")
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=[1, 10, 100, 1_000])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    if any(size_mb <= 0 for size_mb in args.sizes_mb) or args.warmup < 0 or args.steps < 1:
        parser.error("Sizes and steps must be positive, and warmup must be nonnegative.")
    if "LOCAL_RANK" not in os.environ:
        parser.error("Launch with torchrun --standalone --nproc-per-node=<number of GPUs>.")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != int(os.environ["LOCAL_WORLD_SIZE"]):
        parser.error("This benchmark requires all processes on one machine.")
    if torch.cuda.device_count() < world_size:
        parser.error("Each process needs its own visible CUDA GPU.")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=60))

    try:
        rank = dist.get_rank()
        # Check that a sum reaches every rank before timing the larger tensors.
        probe = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(probe)
        expected = world_size * (world_size + 1) // 2
        if probe.item() != expected:
            raise RuntimeError("The all-reduce sanity check returned the wrong sum.")

        if rank == 0:
            print("gpus,size_mb,steps,mean_ms,std_ms", flush=True)

        for size_mb in args.sizes_mb:
            timings = benchmark_size(size_mb, device, args.warmup, args.steps)
            if rank == 0:
                values = timings.tolist()
                mean_ms = statistics.mean(values)
                std_ms = statistics.pstdev(values)
                print(f"{world_size},{size_mb},{args.steps},{mean_ms:.3f},{std_ms:.3f}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
