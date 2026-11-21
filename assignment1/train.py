import argparse
from pathlib import Path
import csv
import json
import math
from time import perf_counter

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.batch import get_batch
from cs336_basics.loss import cross_entropy
from cs336_basics.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.transformer import TransformerLM

def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number

def parse_args() -> argparse.Namespace:
    """读取启动参数，并检查学习率调度等设置。"""
    parser = argparse.ArgumentParser(description="Train a CS336 language model.")

    parser.add_argument("--train-tokens", type=Path, required=True)
    parser.add_argument("--val-tokens", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=positive_int, required=True)
    parser.add_argument("--context-length", type=positive_int, default=128)
    parser.add_argument("--d-model", type=positive_int, default=128)
    parser.add_argument("--num-layers", type=positive_int, default=2)
    parser.add_argument("--num-heads", type=positive_int, default=4)
    parser.add_argument("--d-ff", type=positive_int, default=384)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # max_steps 是训练结束时的总步数，包含检查点里已完成的步数。
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--max-steps", type=positive_int, required=True)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--cosine-cycle-iters", type=positive_int)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-every", type=positive_int, default=10)
    parser.add_argument("--eval-every", type=positive_int, default=100)
    parser.add_argument("--eval-batches", type=positive_int, default=8)
    parser.add_argument("--save-every", type=positive_int, default=100)

    args = parser.parse_args()

    if args.cosine_cycle_iters is None:
        args.cosine_cycle_iters = args.max_steps
    if not 0 <= args.warmup_iters < args.cosine_cycle_iters:
        parser.error("require 0 <= warmup-iters < cosine-cycle-iters")
    if not 0 <= args.min_lr <= args.max_lr:
        parser.error("require 0 <= min-lr <= max-lr")
    if args.max_grad_norm <= 0:
        parser.error("max-grad-norm must be positive")

    return args

def open_tokens(path: Path, context_length: int) -> np.ndarray:
    """打开磁盘上的一维 token ID 数组。"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        tokens = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".bin":
        tokens = np.memmap(path, mode="r", dtype=np.uint16)
    else:
        raise ValueError(f"Expected a .npy or raw uint16 .bin file, got {path}")

    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"{path} must contain a one-dimensional array of token IDs")
    if len(tokens) <= context_length:
        raise ValueError(f"{path} needs at least {context_length + 1} tokens")

    return tokens

@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_tokens: np.ndarray,
    args: argparse.Namespace,
) -> float:
    """用若干验证 batch 估计当前模型的平均 loss。"""
    model.eval()
    total_loss = 0.0

    for _ in range(args.eval_batches):
        x, y = get_batch(
            val_tokens, args.batch_size, args.context_length, args.device
        )
        total_loss += cross_entropy(model(x), y).item()

    model.train()
    return total_loss / args.eval_batches


def main() -> None:
    started_at = perf_counter()
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tokens = open_tokens(args.train_tokens, args.context_length)
    val_tokens = open_tokens(args.val_tokens, args.context_length)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    completed_steps = 0
    if args.resume is not None:
        completed_steps = load_checkpoint(args.resume, model, optimizer)
        print(f"loaded checkpoint at step {completed_steps}", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_suffix = f"from-{completed_steps:08d}"
    config_path = args.checkpoint_dir / f"config-{run_suffix}.json"
    metrics_path = args.checkpoint_dir / f"metrics-{run_suffix}.csv"

    config_path.write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            ["step", "elapsed_s", "tokens_seen", "train_loss", "val_loss", "lr"]
        )

    model.train()

    for it in range(completed_steps, args.max_steps):
        lr = get_lr_cosine_schedule(
            it,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(
            train_tokens, args.batch_size, args.context_length, args.device
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        step = it + 1
        last_step = step == args.max_steps

        log_now = step % args.log_every == 0 or last_step
        eval_now = step % args.eval_every == 0 or last_step
        save_now = step % args.save_every == 0 or last_step

        if log_now or eval_now or save_now:
            train_loss = loss.item()
            val_loss = None

            if eval_now and math.isfinite(train_loss):
                val_loss = evaluate(model, val_tokens, args)

            elapsed_s = perf_counter() - started_at
            tokens_seen = step * args.batch_size * args.context_length

            with metrics_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    [step, elapsed_s, tokens_seen, train_loss, val_loss, lr]
                )

            print(f"step={step} train_loss={train_loss:.4f} lr={lr:.6g}", flush=True)
            if val_loss is not None:
                print(f"step={step} val_loss={val_loss:.4f}", flush=True)

            if not math.isfinite(train_loss) or (
                val_loss is not None and not math.isfinite(val_loss)
            ):
                print(f"non-finite loss at step {step}", flush=True)
                break

        if save_now:
            checkpoint_path = args.checkpoint_dir / f"step-{step:08d}.pt"
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"saved {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()