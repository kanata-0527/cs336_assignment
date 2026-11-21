from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

import torch
import os
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] 
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    if isinstance(out, (str, PathLike)):
        destination = Path(out)
        # 完整写入临时文件后再替换目标文件。
        with TemporaryDirectory(dir=destination.parent) as directory:
            temporary_path = Path(directory) / "checkpoint.pt"
            torch.save(state, temporary_path)
            temporary_path.replace(destination)
    else:
        torch.save(state, out)

def load_checkpoint(
    src: str | PathLike[str] | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]