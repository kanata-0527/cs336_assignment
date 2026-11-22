import argparse
import os
from contextlib import nullcontext
from statistics import mean

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import MODEL_SIZES, synchronized_time
from cs336_systems.ddp import FlatDDP, OverlapDDP
from cs336_systems.fsdp import FSDP
from cs336_systems.naive_ddp import NaiveDDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


def report_memory(phase, model, optimizer, device):
    """Report live allocations and the peak since entering this phase."""
    torch.cuda.synchronize(device)
    mib = 1024**2
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    gradient_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    )
    state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.device == device
    )

    print(
        f"rank={dist.get_rank()} phase={phase} "
        f"allocated_mib={torch.cuda.memory_allocated(device) / mib:.2f} "
        f"peak_mib={torch.cuda.max_memory_allocated(device) / mib:.2f} "
        f"parameters_mib={parameter_bytes / mib:.2f} "
        f"gradients_mib={gradient_bytes / mib:.2f} "
        f"optimizer_state_mib={state_bytes / mib:.2f}",
        flush=True,
    )


def time_train_step(
    model, tokens, targets, optimizer, *, step_only=False, nvtx=False, memory_accounting=False
):
    device = tokens.device

    def mark(name):
        return torch.cuda.nvtx.range(name) if nvtx else nullcontext()

    dist.barrier()
    start = synchronized_time(device)

    with mark("measured_step"):
        optimizer.zero_grad(set_to_none=True)
        if memory_accounting:
            torch.cuda.reset_peak_memory_stats(device)

        with mark("forward"):
            loss = cross_entropy(model(tokens).float(), targets)

        with mark("backward"):
            loss.backward()

        sync_start = None if step_only else synchronized_time(device)

        with mark("gradient_sync"):
            model.sync_gradients()

        sync_s = float("nan")
        if sync_start is not None:
            sync_s = synchronized_time(device) - sync_start

        if memory_accounting:
            report_memory("before_optimizer", model, optimizer, device)
            torch.cuda.reset_peak_memory_stats(device)

        with mark("optimizer"):
            optimizer.step()

        if memory_accounting:
            report_memory("after_optimizer", model, optimizer, device)

        end = synchronized_time(device)

    return end - start, sync_s


def report_timings(timings, device, *, step_only=False, metric="step_ms"):
    local = torch.tensor(timings, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)

    if dist.get_rank() != 0:
        return

    timings_by_rank = [rank_timings.tolist() for rank_timings in gathered]
    slowest_steps = []

    for step_timings in zip(*timings_by_rank):
        # 选出这轮整步最慢的 rank，保留它的两项耗时。
        slowest = max(step_timings, key=lambda pair: pair[0])
        slowest_steps.append(slowest)

    step_s = mean(total for total, _ in slowest_steps)
    print(f"{metric}={step_s * 1_000:.3f}")
    if not step_only:
        sync_s = mean(sync for _, sync in slowest_steps)
        print(f"gradient_sync_ms={sync_s * 1_000:.3f}")
        print(f"gradient_sync_pct={100 * sync_s / step_s:.1f}%")


def measure_forward(model, tokens, args, device):
    timings = []
    for step in range(args.warmup + args.steps):
        measured = step >= args.warmup
        profiling = measured and args.nvtx
        if profiling:
            model.start_forward_wait_timing()
        label = torch.cuda.nvtx.range("measured_forward") if profiling else nullcontext()
        dist.barrier()
        start = synchronized_time(device)
        with label:
            output = model(tokens)
            end = synchronized_time(device)
        del output

        if measured:
            timings.append((end - start, float("nan")))
        if profiling:
            rank = dist.get_rank()
            for layer, wait_ms in model.finish_forward_wait_timing():
                print(f"forward_wait rank={rank} step={step - args.warmup} layer={layer} wait_ms={wait_ms:.6f}")

    report_timings(timings, device, step_only=True, metric="forward_ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODEL_SIZES, default="tiny")
    parser.add_argument("--batch-size", type=int, default=4, help="Global batch size")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--ddp-mode", choices=("naive", "flat", "overlap", "fsdp"), default="naive"
    )
    parser.add_argument("--fsdp-compute-dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--step-only", action="store_true", help="Measure full steps without synchronizing between phases")
    parser.add_argument("--nvtx", action="store_true", help="Mark measured training steps for Nsight Systems")
    parser.add_argument("--optimizer", choices=("adamw", "sharded"), default="adamw")
    parser.add_argument("--forward-only", action="store_true", help="Measure the FSDP training forward pass")
    parser.add_argument("--memory-accounting", action="store_true", help="Measure per-rank GPU memory through the first update")
    args = parser.parse_args()
    step_only = args.step_only or args.ddp_mode in ("overlap", "fsdp")

    if args.ddp_mode == "fsdp" and args.optimizer == "sharded":
        parser.error("Use --optimizer adamw with FSDP. Each rank already owns different parameter shards.")
    if args.forward_only and (args.ddp_mode != "fsdp" or args.memory_accounting):
        parser.error("--forward-only requires FSDP and a separate run from --memory-accounting.")
    if "LOCAL_RANK" not in os.environ:
        parser.error("Launch this script with torchrun.")
    if min(args.batch_size, args.context_length, args.steps) < 1 or args.warmup < 0:
        parser.error("Batch size, context length and steps must be positive; warmup must be nonnegative.")

    world_size = int(os.environ["WORLD_SIZE"])
    if args.batch_size % world_size != 0:
        parser.error("Global batch size must be divisible by the number of ranks.")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    try:
        rank = dist.get_rank()
        local_batch = args.batch_size // world_size
        vocab_size = 128 if args.size == "tiny" else 10_000
        d_model, d_ff, num_layers, num_heads = MODEL_SIZES[args.size]

        torch.manual_seed(0)
        if args.memory_accounting:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        base_model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=args.context_length,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        if args.ddp_mode == "fsdp":
            compute_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.fsdp_compute_dtype]
            model = FSDP(base_model, compute_dtype=compute_dtype, nvtx=args.nvtx, device=device)
        else:
            base_model.to(device)
            ddp_class = {
                "naive": NaiveDDP,
                "flat": FlatDDP,
                "overlap": OverlapDDP,
            }[args.ddp_mode]
            model = ddp_class(base_model)
        if args.forward_only:
            optimizer = None
        elif args.optimizer == "sharded":
            optimizer = ShardedOptimizer(model.parameters(), AdamW)
        else:
            optimizer = AdamW(model.parameters())

        if args.memory_accounting:
            report_memory("after_init", model, optimizer, device)

        # 各 rank 用不同随机种子生成自己的样本。
        generator = torch.Generator(device=device).manual_seed(1_000 + rank)
        sequences = torch.randint(
            vocab_size,
            (local_batch, args.context_length + 1),
            generator=generator,
            device=device,
        )
        tokens = sequences[:, :-1].contiguous()
        targets = sequences[:, 1:].contiguous()

        if rank == 0:
            compute_precision = args.fsdp_compute_dtype if args.ddp_mode == "fsdp" else "fp32"
            print(
                f"size={args.size} gpus={world_size} ddp_mode={args.ddp_mode} "
                f"compute_precision={compute_precision} optimizer={args.optimizer} forward_only={args.forward_only} "
                f"global_batch={args.batch_size} local_batch={local_batch} "
                f"context={args.context_length} warmup={args.warmup} steps={args.steps}",
                flush=True,
            )

        if args.forward_only:
            measure_forward(model, tokens, args, device)
            return

        if args.memory_accounting:
            time_train_step(
                model, tokens, targets, optimizer, step_only=True, memory_accounting=True
            )
            return

        timings = []
        for step in range(args.warmup + args.steps):
            elapsed = time_train_step(
                model, tokens, targets, optimizer,
                step_only=step_only,
                nvtx=args.nvtx and step >= args.warmup,
            )
            if step >= args.warmup:
                timings.append(elapsed)

        report_timings(timings, device, step_only=step_only)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
