from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, output_dir: str, name: str) -> str:
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pt"
    torch.save(state, path)
    return str(path)
