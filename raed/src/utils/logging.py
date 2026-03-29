from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import wandb


def create_logger(cfg: dict[str, Any]) -> wandb.sdk.wandb_run.Run | None:
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None
    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=wandb_cfg.get("project", "raed"),
        group=wandb_cfg.get("group", "stage_a"),
        name=cfg.get("run_name", "raed_stage_a"),
        dir=str(output_dir),
        config=cfg,
    )


def log_metrics(run: wandb.sdk.wandb_run.Run | None, metrics: dict[str, float], step: int | None = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def dump_metrics_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
