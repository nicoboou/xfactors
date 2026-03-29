from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as f

from raed.src.eval.common import compute_clean_anchor_t, load_stage_b2_diffusion, save_table_csv
from raed.src.losses import VGGPerceptualLoss
from raed.src.utils.logging import dump_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage B2 diffusion refinement")
    parser.add_argument("--config", type=str, default="raed/configs/stage_b2_diffusion.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="raed/outputs/eval_diffusion_refinement")
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


def _decode_option2(decoder, decoder_mode: str, decoder_input_mode: str, stage_a_model, z: torch.Tensor, shallow: torch.Tensor | None):
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
    cfg, stage_a_model, encoder, diffusion_model, decoder, decoder_mode, decoder_input_mode, val_loader, device = load_stage_b2_diffusion(
        args.config,
        args.overrides,
        args.checkpoint,
    )
    option = cfg["diffusion"].get("option", "option1")
    image_size = int(cfg["data"]["image_size"])
    refine_steps = int(cfg["inference"].get("refine_steps", 20))
    perceptual = VGGPerceptualLoss().to(device).eval()

    anchor_t = None
    if option == "option2":
        anchor_t = compute_clean_anchor_t(stage_a_model, encoder, val_loader, device)

    totals = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_proxy": 0.0, "DINO_similarity": 0.0, "count": 0}
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["is_degraded"].to(device)
            deg_mask = y == 1
            if not deg_mask.any():
                continue
            x_bad = x[deg_mask]

            deep_bad = encoder.forward_deep(x_bad)
            out_bad = stage_a_model(deep_bad)
            if option == "option1":
                x_restored = diffusion_model.sample(out_bad["s"], image_size=image_size, steps=refine_steps)
            else:
                t_anchor = anchor_t.repeat(out_bad["s"].size(0), 1)
                z_init = torch.cat([out_bad["s"], t_anchor], dim=-1)
                z_refined = diffusion_model.refine(z_init, refine_steps=refine_steps)
                shallow_bad = encoder.forward_shallow(x_bad)
                x_restored = _decode_option2(decoder, decoder_mode, decoder_input_mode, stage_a_model, z_refined, shallow_bad)

            deep_restored = encoder.forward_deep(x_restored)
            totals["PSNR"] += float(batch_psnr(x_restored, x_bad))
            totals["SSIM"] += float(batch_ssim_global(x_restored, x_bad))
            totals["LPIPS_proxy"] += float(perceptual(x_restored, x_bad))
            totals["DINO_similarity"] += float(f.cosine_similarity(deep_restored.mean(dim=1), deep_bad.mean(dim=1), dim=-1).mean())
            totals["count"] += 1

    n = max(int(totals["count"]), 1)
    report = {
        "option": option,
        "PSNR": totals["PSNR"] / n,
        "SSIM": totals["SSIM"] / n,
        "LPIPS_proxy": totals["LPIPS_proxy"] / n,
        "DINO_similarity": totals["DINO_similarity"] / n,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_metrics_json(out_dir / "diffusion_refinement_metrics.json", report)
    dump_metrics_json(out_dir / f"{option}_metrics.json", report)
    save_table_csv(str(out_dir / "diffusion_refinement_metrics.csv"), [report])
    print(report)


if __name__ == "__main__":
    main()
