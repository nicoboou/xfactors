from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from raed.src.data import build_celeba_weak_dataloaders
from raed.src.losses import deep_reconstruction_loss, kl_standard_normal, supervised_info_nce
from raed.src.models import DinoReconstructor, FrozenDinoEncoder, VariationalFactorizer
from raed.src.utils import apply_overrides, create_logger, load_config, log_metrics, save_checkpoint, seed_everything


class StageAModel(nn.Module):
    def __init__(self, cfg: dict, in_dim: int):
        super().__init__()
        model_cfg = cfg["model"]
        self.factorizer = VariationalFactorizer(
            in_dim=in_dim,
            latent_dim_s=model_cfg["latent_dim_s"],
            latent_dim_t=model_cfg["latent_dim_t"],
            hidden_dim=model_cfg["hidden_dim"],
        )
        self.reconstructor = DinoReconstructor(
            latent_dim_s=model_cfg["latent_dim_s"],
            latent_dim_t=model_cfg["latent_dim_t"],
            out_dim=in_dim,
            hidden_dim=model_cfg["hidden_dim"],
        )
        self.t_projector = nn.Sequential(
            nn.Linear(model_cfg["latent_dim_t"], model_cfg["proj_dim"]),
            nn.GELU(),
            nn.Linear(model_cfg["proj_dim"], model_cfg["proj_dim"]),
        )

    def forward(self, deep_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        fac = self.factorizer(deep_tokens)
        h_hat = self.reconstructor(fac["s"], fac["t"])
        t_proj = self.t_projector(fac["t"])
        return {**fac, "h_hat": h_hat, "t_proj": t_proj}


def _stage_a_loss(cfg: dict, outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_cfg = cfg["loss"]
    rec_total, rec_mse, rec_cos = deep_reconstruction_loss(
        outputs["h_hat"],
        outputs["pooled"],
        lambda_mse=loss_cfg["lambda_mse"],
        lambda_cos=loss_cfg["lambda_cos"],
    )
    l_nce = supervised_info_nce(outputs["t_proj"], labels, temperature=loss_cfg["temperature"])
    kl_s = kl_standard_normal(outputs["mu_s"], outputs["logvar_s"])
    kl_t = kl_standard_normal(outputs["mu_t"], outputs["logvar_t"])
    total = rec_total + loss_cfg["lambda_nce"] * l_nce + loss_cfg["lambda_kl_s"] * kl_s + loss_cfg["lambda_kl_t"] * kl_t
    metrics = {
        "L_rec_deep": rec_total.detach(),
        "L_rec_mse": rec_mse.detach(),
        "L_rec_cos": rec_cos.detach(),
        "L_nce": l_nce.detach(),
        "KL_s": kl_s.detach(),
        "KL_t": kl_t.detach(),
        "loss_total": total.detach(),
    }
    return total, metrics


@torch.no_grad()
def validate(cfg: dict, model: StageAModel, encoder: FrozenDinoEncoder, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    running = {"loss_total": 0.0, "L_rec_deep": 0.0, "L_nce": 0.0, "KL_s": 0.0, "KL_t": 0.0}
    batches = 0
    for batch in tqdm(val_loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch["is_degraded"].to(device)
        deep_tokens = encoder.forward_deep(x)
        out = model(deep_tokens)
        _, metrics = _stage_a_loss(cfg, out, y)
        for key in running:
            running[key] += float(metrics[key])
        batches += 1
    return {f"val/{k}": v / max(batches, 1) for k, v in running.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAED Stage A")
    parser.add_argument("--config", type=str, default="raed/configs/stage_a_base.yaml")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    seed_everything(int(cfg["seed"]))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run = create_logger(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_celeba_weak_dataloaders(cfg)
    encoder = FrozenDinoEncoder(**cfg["encoder"]).to(device)
    encoder.eval()

    first_batch = next(iter(data.train))
    with torch.no_grad():
        deep_tokens = encoder.forward_deep(first_batch["image"].to(device))
        in_dim = deep_tokens.shape[-1]

    model = StageAModel(cfg, in_dim=in_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True) and device.type == "cuda"))

    best_val = float("inf")
    global_step = 0
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        optim.zero_grad(set_to_none=True)
        running = {"L_rec_deep": 0.0, "L_nce": 0.0, "KL_s": 0.0, "KL_t": 0.0, "loss_total": 0.0}
        batches = 0
        for step, batch in enumerate(tqdm(data.train, desc=f"train {epoch}", leave=False), start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["is_degraded"].to(device, non_blocking=True)
            with torch.no_grad():
                deep_tokens = encoder.forward_deep(x)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(deep_tokens)
                loss, metrics = _stage_a_loss(cfg, outputs, y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            if step % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("max_grad_norm", 1.0))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            for key in running:
                running[key] += float(metrics[key])
            batches += 1
            global_step += 1
            if global_step % int(cfg["train"].get("log_every", 20)) == 0:
                log_metrics(run, {f"train/{k}": running[k] / batches for k in running}, step=global_step)

        train_payload = {f"train/{k}": v / max(batches, 1) for k, v in running.items()}
        log_metrics(run, train_payload, step=global_step)

        val_metrics = validate(cfg, model, encoder, data.val, device)
        val_score = val_metrics["val/loss_total"]
        log_metrics(run, val_metrics, step=global_step)

        state = {
            "epoch": epoch,
            "config": cfg,
            "model": model.state_dict(),
            "encoder": cfg["encoder"],
            "best_val": best_val,
        }
        save_checkpoint(state, str(out_dir), "last")
        if val_score < best_val:
            best_val = val_score
            save_checkpoint(state, str(out_dir), "best")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
