from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as f
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from raed.src.data import build_weak_dataloaders
from raed.src.losses import VGGPerceptualLoss
from raed.src.models import DinoTokPixelDecoder, FrozenDinoEncoder, PlainPixelDecoder
from raed.src.train.train_stage_a import StageAModel
from raed.src.utils import (
    apply_overrides,
    cleanup_distributed,
    create_logger,
    ddp_wrap,
    init_distributed,
    is_main_process,
    load_config,
    log_metrics,
    reduce_metrics,
    save_checkpoint,
    seed_everything,
    set_cuda_visible_devices,
    state_dict_for_save,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RAED Stage B decoder")
    parser.add_argument("--config", type=str, default="raed/configs/stage_b_decoder.yaml")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def batch_psnr(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = f.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps)).mean()


def batch_ssim_global(x_hat: torch.Tensor, x: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = x.mean(dim=(1, 2, 3))
    mu_y = x_hat.mean(dim=(1, 2, 3))
    sigma_x = x.var(dim=(1, 2, 3), unbiased=False)
    sigma_y = x_hat.var(dim=(1, 2, 3), unbiased=False)
    cov = ((x - mu_x[:, None, None, None]) * (x_hat - mu_y[:, None, None, None])).mean(dim=(1, 2, 3))
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.mean()


def _build_decoder(cfg: dict, input_dim: int, shallow_dim: int):
    mode = cfg["model"].get("decoder_mode", "plain")
    if mode == "plain":
        return PlainPixelDecoder(input_dim)
    if mode == "dinotok":
        return DinoTokPixelDecoder(deep_dim=input_dim, shallow_dim=shallow_dim, fused_dim=cfg["model"].get("feature_dim", input_dim))
    raise ValueError(f"Unknown decoder_mode={mode}")


def _decoder_forward(decoder, decoder_mode: str, deep_input: torch.Tensor, shallow: torch.Tensor | None) -> torch.Tensor:
    if decoder_mode == "dinotok":
        return decoder(deep_input, shallow)
    return decoder(deep_input, None)


def _prepare_latent_input(input_mode: str, stage_a_out: dict[str, torch.Tensor]) -> torch.Tensor:
    if input_mode == "st":
        return torch.cat([stage_a_out["s"], stage_a_out["t"]], dim=-1)
    if input_mode == "h_hat":
        return stage_a_out["h_hat"]
    raise ValueError(f"Unknown input_mode={input_mode}")


def validate(
    cfg: dict,
    stage_a_model: StageAModel,
    encoder: FrozenDinoEncoder,
    decoder,
    perceptual,
    val_loader,
    device: torch.device,
) -> dict[str, float]:
    stage_a_model.eval()
    decoder.eval()
    running = {
        "loss_total": 0.0,
        "L1": 0.0,
        "L_perceptual": 0.0,
        "L_sem": 0.0,
        "PSNR": 0.0,
        "SSIM": 0.0,
        "DINO_similarity": 0.0,
    }
    batches = 0
    decoder_mode = cfg["model"].get("decoder_mode", "plain")
    input_mode = cfg["model"].get("input_mode", "st")
    for batch in tqdm(val_loader, desc="val_b", leave=False):
        x = batch["image"].to(device)
        with torch.no_grad():
            deep = encoder.forward_deep(x)
            shallow = encoder.forward_shallow(x)
            stage_a_out = stage_a_model(deep)
            decoder_in = _prepare_latent_input(input_mode, stage_a_out)
            x_hat = _decoder_forward(decoder, decoder_mode, decoder_in, shallow)
            l1 = f.l1_loss(x_hat, x)
            l_perc = perceptual(x_hat, x)
            deep_hat = encoder.forward_deep(x_hat)
            sem = 1.0 - f.cosine_similarity(deep_hat.mean(dim=1), deep.mean(dim=1), dim=-1).mean()
            total = cfg["loss"]["lambda_l1"] * l1 + cfg["loss"]["lambda_lpips"] * l_perc + cfg["loss"].get("lambda_sem", 0.0) * sem
            dino_sim = f.cosine_similarity(deep_hat.mean(dim=1), deep.mean(dim=1), dim=-1).mean()
            running["loss_total"] += float(total)
            running["L1"] += float(l1)
            running["L_perceptual"] += float(l_perc)
            running["L_sem"] += float(sem)
            running["PSNR"] += float(batch_psnr(x_hat, x))
            running["SSIM"] += float(batch_ssim_global(x_hat, x))
            running["DINO_similarity"] += float(dino_sim)
        batches += 1
    return {f"val/{k}": v / max(batches, 1) for k, v in running.items()}


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_cuda_visible_devices(cfg.get("runtime", {}).get("gpu_ids", []))
    device, rank = init_distributed(cfg)
    seed_everything(int(cfg["seed"]) + rank)

    out_dir = Path(cfg["output_dir"])
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    run = create_logger(cfg) if is_main_process() else None

    data = build_weak_dataloaders(cfg)

    stage_a_path = cfg["stage_a_checkpoint"]
    stage_a_state = torch.load(stage_a_path, map_location="cpu")
    stage_a_cfg = stage_a_state["config"]

    encoder = FrozenDinoEncoder(**stage_a_cfg["encoder"]).to(device)
    encoder.eval()

    first_batch = next(iter(data.train))
    with torch.no_grad():
        deep = encoder.forward_deep(first_batch["image"].to(device))
        in_dim = deep.shape[-1]
        shallow = encoder.forward_shallow(first_batch["image"].to(device))
        shallow_dim = in_dim if shallow is None else shallow.shape[-1]

    stage_a_model = StageAModel(stage_a_cfg, in_dim=in_dim).to(device)
    stage_a_model.load_state_dict(stage_a_state["model"])
    if cfg["model"].get("freeze_stage_a", True):
        stage_a_model.eval()
        for param in stage_a_model.parameters():
            param.requires_grad = False

    input_dim = cfg["model"]["latent_dim_s"] + cfg["model"]["latent_dim_t"] if cfg["model"].get("input_mode", "st") == "st" else in_dim
    decoder = _build_decoder(cfg, input_dim=input_dim, shallow_dim=shallow_dim).to(device)
    decoder = ddp_wrap(decoder, device=device, find_unused_parameters=bool(cfg.get("runtime", {}).get("find_unused_parameters", False)))
    perceptual = VGGPerceptualLoss().to(device).eval()

    optim = torch.optim.AdamW(decoder.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True) and device.type == "cuda"))

    best_val = float("inf")
    global_step = 0
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    decoder_mode = cfg["model"].get("decoder_mode", "plain")
    input_mode = cfg["model"].get("input_mode", "st")

    for epoch in range(int(cfg["train"]["epochs"])):
        if data.train_sampler is not None:
            data.train_sampler.set_epoch(epoch)
        decoder.train()
        if cfg["model"].get("freeze_stage_a", True):
            stage_a_model.eval()
        else:
            stage_a_model.train()
        optim.zero_grad(set_to_none=True)
        running = {"loss_total": 0.0, "L1": 0.0, "L_perceptual": 0.0, "L_sem": 0.0, "PSNR": 0.0, "SSIM": 0.0, "DINO_similarity": 0.0}
        batches = 0

        for step, batch in enumerate(tqdm(data.train, desc=f"train_b {epoch}", leave=False), start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["is_degraded"].to(device, non_blocking=True)

            with torch.no_grad():
                deep = encoder.forward_deep(x)
                shallow = encoder.forward_shallow(x)

            if cfg["model"].get("freeze_stage_a", True):
                with torch.no_grad():
                    stage_a_out = stage_a_model(deep)
            else:
                stage_a_out = stage_a_model(deep)

            decoder_in = _prepare_latent_input(input_mode, stage_a_out)

            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                x_hat = _decoder_forward(decoder, decoder_mode, decoder_in, shallow)
                l1 = f.l1_loss(x_hat, x)
                l_perc = perceptual(x_hat, x)
                deep_hat = encoder.forward_deep(x_hat)
                sem = 1.0 - f.cosine_similarity(deep_hat.mean(dim=1), deep.mean(dim=1), dim=-1).mean()
                loss = cfg["loss"]["lambda_l1"] * l1 + cfg["loss"]["lambda_lpips"] * l_perc + cfg["loss"].get("lambda_sem", 0.0) * sem

                clean_mask = y == 0
                deg_mask = y == 1
                if deg_mask.any():
                    if clean_mask.any():
                        t_anchor = stage_a_out["t"][clean_mask].mean(dim=0, keepdim=True)
                    else:
                        t_anchor = torch.zeros_like(stage_a_out["t"][:1])
                    s_deg = stage_a_out["s"][deg_mask]
                    t_deg_anchor = t_anchor.repeat(s_deg.size(0), 1)
                    stage_b_input_deg = torch.cat([s_deg, t_deg_anchor], dim=-1) if input_mode == "st" else stage_a_model.reconstructor(s_deg, t_deg_anchor)
                    shallow_deg = None if shallow is None else shallow[deg_mask]
                    x_cleanish = _decoder_forward(decoder, decoder_mode, stage_b_input_deg, shallow_deg)
                    cleanish_reg = f.l1_loss(x_cleanish, x[deg_mask])
                    loss = loss + 0.05 * cleanish_reg

                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()
            if step % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg["train"].get("max_grad_norm", 1.0))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            dino_sim = f.cosine_similarity(deep_hat.mean(dim=1), deep.mean(dim=1), dim=-1).mean()
            running["loss_total"] += float(loss.detach())
            running["L1"] += float(l1.detach())
            running["L_perceptual"] += float(l_perc.detach())
            running["L_sem"] += float(sem.detach())
            running["PSNR"] += float(batch_psnr(x_hat.detach(), x))
            running["SSIM"] += float(batch_ssim_global(x_hat.detach(), x))
            running["DINO_similarity"] += float(dino_sim.detach())
            batches += 1
            global_step += 1

            if global_step % int(cfg["train"].get("log_every", 20)) == 0:
                payload = {f"train/{k}": v / max(batches, 1) for k, v in running.items()}
                payload = reduce_metrics(payload, device)
                if is_main_process():
                    log_metrics(run, payload, step=global_step)

        train_payload = {f"train/{k}": v / max(batches, 1) for k, v in running.items()}
        train_payload = reduce_metrics(train_payload, device)
        if is_main_process():
            log_metrics(run, train_payload, step=global_step)

        val_metrics = validate(cfg, stage_a_model, encoder, decoder, perceptual, data.val, device)
        val_metrics = reduce_metrics(val_metrics, device)
        if is_main_process():
            log_metrics(run, val_metrics, step=global_step)
        val_score = val_metrics["val/loss_total"]

        state = {
            "epoch": epoch,
            "config": cfg,
            "decoder": state_dict_for_save(decoder),
            "stage_a_checkpoint": stage_a_path,
            "best_val": best_val,
        }
        if is_main_process():
            save_checkpoint(state, str(out_dir), "last")
            if val_score < best_val:
                best_val = val_score
                save_checkpoint(state, str(out_dir), "best")

    if is_main_process() and run is not None:
        run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
