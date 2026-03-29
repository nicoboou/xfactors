from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as f
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from raed.src.data import build_weak_dataloaders
from raed.src.models import (
    DinoTokPixelDecoder,
    FrozenDinoEncoder,
    LatentRAEDDiffusion,
    PlainPixelDecoder,
    SemanticConditionedPixelDiffusion,
)
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
    parser = argparse.ArgumentParser(description="Train RAED Stage B2 diffusion")
    parser.add_argument("--config", type=str, default="raed/configs/stage_b2_diffusion.yaml")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def _load_stage_a_and_encoder(cfg: dict, device: torch.device):
    stage_a_state = torch.load(cfg["stage_a_checkpoint"], map_location="cpu")
    stage_a_cfg = stage_a_state["config"]
    encoder = FrozenDinoEncoder(**stage_a_cfg["encoder"]).to(device)
    encoder.eval()
    return stage_a_cfg, stage_a_state, encoder


def _build_stage_b_decoder(decoder_cfg: dict, state: dict, input_dim: int, shallow_dim: int, device: torch.device):
    decoder_mode = decoder_cfg["model"].get("decoder_mode", "plain")
    if decoder_mode == "plain":
        decoder = PlainPixelDecoder(input_dim).to(device)
    else:
        decoder = DinoTokPixelDecoder(
            deep_dim=input_dim,
            shallow_dim=shallow_dim,
            fused_dim=decoder_cfg["model"].get("feature_dim", input_dim),
        ).to(device)
    decoder.load_state_dict(state["decoder"])
    return decoder, decoder_mode, decoder_cfg["model"].get("input_mode", "st")


def _decode_option2(
    decoder,
    decoder_mode: str,
    decoder_input_mode: str,
    stage_a_model: StageAModel,
    z: torch.Tensor,
    shallow: torch.Tensor | None,
) -> torch.Tensor:
    s_dim = stage_a_model.factorizer.mu_s.out_features
    s, t = z[:, :s_dim], z[:, s_dim:]
    if decoder_input_mode == "st":
        dec_in = torch.cat([s, t], dim=-1)
    else:
        dec_in = stage_a_model.reconstructor(s, t)
    if decoder_mode == "dinotok":
        return decoder(dec_in, shallow)
    return decoder(dec_in, None)


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

    stage_a_cfg, stage_a_state, encoder = _load_stage_a_and_encoder(cfg, device)
    first_batch = next(iter(data.train))
    first_image = first_batch["image"].to(device)
    with torch.no_grad():
        deep = encoder.forward_deep(first_image)
        in_dim = deep.shape[-1]
        shallow = encoder.forward_shallow(first_image)
        shallow_dim = in_dim if shallow is None else shallow.shape[-1]

    stage_a_model = StageAModel(stage_a_cfg, in_dim=in_dim).to(device)
    stage_a_model.load_state_dict(stage_a_state["model"])
    if cfg["model"].get("freeze_stage_a", True):
        stage_a_model.eval()
        for p in stage_a_model.parameters():
            p.requires_grad = False

    option = cfg["diffusion"].get("option", "option1")
    latent_dim_s = int(cfg["model"]["latent_dim_s"])
    latent_dim_t = int(cfg["model"]["latent_dim_t"])
    num_steps = int(cfg["diffusion"]["num_steps"])
    beta_schedule = cfg["diffusion"].get("beta_schedule", "linear")
    hidden_dim = int(cfg["diffusion"].get("denoiser_hidden_dim", 256))
    depth = int(cfg["diffusion"].get("denoiser_depth", 4))

    decoder = None
    decoder_mode = "plain"
    decoder_input_mode = "st"
    if option == "option1":
        diffusion_model = SemanticConditionedPixelDiffusion(
            s_dim=latent_dim_s,
            num_steps=num_steps,
            schedule=beta_schedule,
            hidden_dim=hidden_dim,
        ).to(device)
    elif option == "option2":
        diffusion_model = LatentRAEDDiffusion(
            z_dim=latent_dim_s + latent_dim_t,
            num_steps=num_steps,
            schedule=beta_schedule,
            hidden_dim=hidden_dim,
            depth=depth,
        ).to(device)
        decoder_state = torch.load(cfg["stage_b_decoder_checkpoint"], map_location="cpu")
        decoder_cfg = decoder_state["config"]
        input_dim = latent_dim_s + latent_dim_t if decoder_cfg["model"].get("input_mode", "st") == "st" else in_dim
        decoder, decoder_mode, decoder_input_mode = _build_stage_b_decoder(decoder_cfg, decoder_state, input_dim, shallow_dim, device)
        if cfg["model"].get("freeze_decoder", True):
            decoder.eval()
            for p in decoder.parameters():
                p.requires_grad = False
    else:
        raise ValueError(f"Unknown diffusion.option={option}")

    diffusion_model = ddp_wrap(
        diffusion_model,
        device=device,
        find_unused_parameters=bool(cfg.get("runtime", {}).get("find_unused_parameters", False)),
    )

    optim = torch.optim.AdamW(diffusion_model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True) and device.type == "cuda"))

    best_val = float("inf")
    global_step = 0
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    refine_steps = int(cfg["inference"].get("refine_steps", 20))

    for epoch in range(int(cfg["train"]["epochs"])):
        if data.train_sampler is not None:
            data.train_sampler.set_epoch(epoch)
        diffusion_model.train()
        optim.zero_grad(set_to_none=True)
        running = {"L_diffusion": 0.0, "preview_metric": 0.0}
        batches = 0
        for step, batch in enumerate(tqdm(data.train, desc=f"train_b2 {epoch}", leave=False), start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["is_degraded"].to(device, non_blocking=True)
            clean_mask = y == 0
            if not clean_mask.any():
                continue
            x_clean = x[clean_mask]

            with torch.no_grad():
                deep = encoder.forward_deep(x_clean)
                shallow_clean = encoder.forward_shallow(x_clean)
                stage_a_out = stage_a_model(deep)

            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                if option == "option1":
                    loss = diffusion_model.loss(x_clean, stage_a_out["s"])
                    preview = loss.detach()
                else:
                    z_clean = torch.cat([stage_a_out["s"], stage_a_out["t"]], dim=-1)
                    loss = diffusion_model.loss(z_clean)
                    z_refined = diffusion_model.refine(z_clean, refine_steps=refine_steps)
                    x_hat = _decode_option2(decoder, decoder_mode, decoder_input_mode, stage_a_model, z_refined, shallow_clean)
                    preview = f.l1_loss(x_hat, x_clean).detach()

                loss = cfg["loss"].get("lambda_diffusion", 1.0) * loss
                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()
            if step % grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), cfg["train"].get("max_grad_norm", 1.0))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running["L_diffusion"] += float(loss.detach())
            running["preview_metric"] += float(preview)
            batches += 1
            global_step += 1

            if global_step % int(cfg["train"].get("log_every", 20)) == 0:
                payload = {
                    "train/L_diffusion": running["L_diffusion"] / max(batches, 1),
                    "train/preview_metric": running["preview_metric"] / max(batches, 1),
                }
                payload = reduce_metrics(payload, device)
                if is_main_process():
                    log_metrics(run, payload, step=global_step)

        train_payload = {
            "train/L_diffusion": running["L_diffusion"] / max(batches, 1),
            "train/preview_metric": running["preview_metric"] / max(batches, 1),
        }
        train_payload = reduce_metrics(train_payload, device)
        if is_main_process():
            log_metrics(run, train_payload, step=global_step)
        val_score = train_payload["train/L_diffusion"]

        state = {
            "epoch": epoch,
            "config": cfg,
            "diffusion_model": state_dict_for_save(diffusion_model),
            "stage_a_checkpoint": cfg["stage_a_checkpoint"],
            "stage_b_decoder_checkpoint": cfg.get("stage_b_decoder_checkpoint", None),
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
