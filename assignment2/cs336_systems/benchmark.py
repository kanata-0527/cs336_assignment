import argparse
from statistics import mean, stdev
from timeit import default_timer

import torch
import os
from contextlib import contextmanager, nullcontext

from torch.cuda import profiler
# from cs336_basics.model import BasicsTransformerLM
from cs336_systems.activation_checkpointing import CheckpointedTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


# 每项依次是 d_model、d_ff、num_layers、num_heads。
# tiny 用于快速验证，其余是讲义 §2.1.2 的配置。
MODEL_SIZES = {
    "tiny": (64, 128, 1, 4),
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
    "10b": (4608, 12288, 50, 36),
}


def synchronized_time(device):
    """在 CUDA 上先等 GPU 完成工作，再读取时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return default_timer()

@contextmanager
def record_cuda_memory(device, snapshot_path):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    try:
        yield
        torch.cuda.synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.memory._dump_snapshot(snapshot_path)
        print(f"peak_allocated={peak_mib:.2f} MiB")
        print(f"saved memory snapshot: {snapshot_path}")
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

def run_step(model, tokens, targets, optimizer, mode, breakdown, precision="fp32"):
    """执行一次指定模式，返回各段耗时，单位为秒。"""
    device = tokens.device
    needs_backward = mode != "forward"

    # 训练模式下，每轮从空梯度开始，避免跨轮累积。
    if needs_backward:
        optimizer.zero_grad(set_to_none=True)

    start = synchronized_time(device)
    forward_end = None
    backward_end = None

    # 只测前向时关闭梯度记录，其余模式保留反向所需的信息。
    precision_context = nullcontext()
    if precision == "bf16":
        precision_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    with torch.set_grad_enabled(needs_backward), precision_context:
        logits = model(tokens)
        if needs_backward:
            loss = cross_entropy(logits.float(), targets)

    if needs_backward:
        if breakdown:
            forward_end = synchronized_time(device)

        loss.backward()

        if mode == "train":
            if breakdown:
                backward_end = synchronized_time(device)
            optimizer.step()

    end = synchronized_time(device)
    timings = {"step": end - start}

    if forward_end is not None:
        timings["forward_and_loss"] = forward_end - start

        if backward_end is None:
            timings["backward"] = end - forward_end
        else:
            timings["backward"] = backward_end - forward_end
            timings["optimizer"] = end - backward_end

    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODEL_SIZES, default="tiny")
    parser.add_argument("--device", default="cpu", help="例如 cpu、cuda 或 cuda:0")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "train"),
        default="forward",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="分别测量前向加 loss、反向和 optimizer 的耗时",
    )
    parser.add_argument("--memory-snapshot", help="保存 CUDA 显存记录的 .pickle 路径")
    parser.add_argument(
    "--checkpoint-block-size",
    type=int,
    default=0,
    help="每个 checkpoint 包含的 Transformer 层数，0 表示关闭",
    )
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile")
    args = parser.parse_args()

    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup 必须大于等于 0，--steps 必须大于等于 1")

    tiny = args.size == "tiny"
    vocab_size = 128 if tiny else 10_000
    batch_size = args.batch_size
    context_length = args.context_length

    if batch_size is None:
        batch_size = 1 if tiny else 4
    if context_length is None:
        context_length = 8 if tiny else 512
    if batch_size < 1 or context_length < 1:
        parser.error("--batch-size 和 --context-length 必须为正整数")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前 Python 环境的 PyTorch 无法使用 CUDA")
    if args.memory_snapshot and device.type != "cuda":
        parser.error("--memory-snapshot 需要使用 CUDA")

    torch.manual_seed(0)
    d_model, d_ff, num_layers, num_heads = MODEL_SIZES[args.size]

    # 模型、优化器和随机数据只创建一次，准备时间不算进 benchmark。
    model = CheckpointedTransformerLM(
    checkpoint_block_size=args.checkpoint_block_size,
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    d_ff=d_ff,
    num_layers=num_layers,
    num_heads=num_heads,
    ).to(device)
    model.train(args.mode != "forward")
    if args.compile:
        model = torch.compile(model, dynamic=False)
    optimizer = AdamW(model.parameters())

    shape = (batch_size, context_length)
    tokens = torch.randint(vocab_size, shape, device=device)
    targets = torch.randint(vocab_size, shape, device=device)

    print(
        f"device={device} size={args.size} mode={args.mode} "
        f"compiled={args.compile} precision={args.precision} "
        f"batch={batch_size} context={context_length} "
        f"warmup={args.warmup} steps={args.steps}",
        flush=True,
    )

    profile_enabled = device.type == "cuda" and os.getenv("CS336_NSYS") == "1"

    measurements = {}
    for step in range(args.warmup + args.steps):
        capture = profile_enabled and step == args.warmup
        recording = profiler.profile() if capture else nullcontext()
        label = torch.cuda.nvtx.range("measured_step") if capture else nullcontext()

        memory_recording = nullcontext()
        if args.memory_snapshot and step == args.warmup:
            memory_recording = record_cuda_memory(device, args.memory_snapshot)

        with memory_recording, recording, label:
            timings = run_step(
                model, tokens, targets, optimizer, args.mode, args.breakdown,
                args.precision,
            )

        if step >= args.warmup:
            for name, seconds in timings.items():
                measurements.setdefault(name, []).append(seconds * 1000)

    for name, samples in measurements.items():
        std = f"{stdev(samples):.3f} ms" if len(samples) > 1 else "n/a"
        print(f"{name}: mean={mean(samples):.3f} ms, std={std}")


if __name__ == "__main__":
    main()
