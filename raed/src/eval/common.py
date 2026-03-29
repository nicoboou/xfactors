from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from raed.src.data import build_celeba_weak_dataloaders
from raed.src.models import DinoTokPixelDecoder, FrozenDinoEncoder, PlainPixelDecoder
from raed.src.train.train_stage_a import StageAModel
from raed.src.utils import apply_overrides, load_config


@torch.no_grad()
def load_stage_a(cfg_path: str, overrides: list[str], checkpoint: str | None):
    cfg = apply_overrides(load_config(cfg_path), overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = checkpoint or str(Path(cfg["output_dir"]) / "checkpoints" / "best.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state.get("config", cfg)

    data = build_celeba_weak_dataloaders(cfg)
    encoder = FrozenDinoEncoder(**cfg["encoder"]).to(device)
    encoder.eval()
    batch = next(iter(data.val))
    deep_tokens = encoder.forward_deep(batch["image"].to(device))
    model = StageAModel(cfg, in_dim=deep_tokens.shape[-1]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return cfg, model, encoder, data.val, device


@torch.no_grad()
def collect_latents(model, encoder, loader, device):
    s_all, t_all, y_all = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["is_degraded"].to(device)
        deep_tokens = encoder.forward_deep(x)
        out = model(deep_tokens)
        s_all.append(out["s"].cpu())
        t_all.append(out["t"].cpu())
        y_all.append(y.cpu())
    s = torch.cat(s_all).numpy()
    t = torch.cat(t_all).numpy()
    y = torch.cat(y_all).numpy()
    return s, t, y


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table_csv(path: str, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    ensure_dir(str(Path(path).parent))
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row[k]) for k in keys) + "\n")


def to_numpy(x: torch.Tensor | np.ndarray):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


@torch.no_grad()
def compute_clean_anchor_t(model: StageAModel, encoder: FrozenDinoEncoder, loader, device: torch.device) -> torch.Tensor:
    acc = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["is_degraded"].to(device)
        mask = y == 0
        if not mask.any():
            continue
        deep = encoder.forward_deep(x)
        out = model(deep)
        acc.append(out["t"][mask].detach())
    if not acc:
        return torch.zeros((1, model.factorizer.mu_t.out_features), device=device)
    return torch.cat(acc, dim=0).mean(dim=0, keepdim=True)


@torch.no_grad()
def load_stage_b_decoder(cfg_path: str, overrides: list[str], checkpoint: str | None):
    cfg = apply_overrides(load_config(cfg_path), overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_ckpt = checkpoint or str(Path(cfg["output_dir"]) / "checkpoints" / "best.pt")
    decoder_state = torch.load(decoder_ckpt, map_location="cpu")
    cfg = decoder_state.get("config", cfg)

    stage_a_state = torch.load(cfg["stage_a_checkpoint"], map_location="cpu")
    stage_a_cfg = stage_a_state["config"]
    data = build_celeba_weak_dataloaders(cfg)

    encoder = FrozenDinoEncoder(**stage_a_cfg["encoder"]).to(device)
    encoder.eval()
    batch = next(iter(data.val))
    deep_tokens = encoder.forward_deep(batch["image"].to(device))
    in_dim = deep_tokens.shape[-1]
    stage_a_model = StageAModel(stage_a_cfg, in_dim=in_dim).to(device)
    stage_a_model.load_state_dict(stage_a_state["model"])
    stage_a_model.eval()

    input_dim = cfg["model"]["latent_dim_s"] + cfg["model"]["latent_dim_t"] if cfg["model"].get("input_mode", "st") == "st" else in_dim
    shallow = encoder.forward_shallow(batch["image"].to(device))
    shallow_dim = in_dim if shallow is None else shallow.shape[-1]
    if cfg["model"].get("decoder_mode", "plain") == "plain":
        decoder = PlainPixelDecoder(input_dim).to(device)
    else:
        decoder = DinoTokPixelDecoder(
            deep_dim=input_dim,
            shallow_dim=shallow_dim,
            fused_dim=cfg["model"].get("feature_dim", input_dim),
        ).to(device)
    decoder.load_state_dict(decoder_state["decoder"])
    decoder.eval()

    return cfg, stage_a_model, encoder, decoder, data.val, device
