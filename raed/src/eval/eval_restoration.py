from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as f

from raed.src.eval.common import compute_clean_anchor_t, load_stage_b_decoder, save_table_csv
from raed.src.losses import VGGPerceptualLoss
from raed.src.utils.logging import dump_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage B clean-anchor restoration")
    parser.add_argument("--config", type=str, default="raed/configs/stage_b_decoder.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="raed/outputs/eval_restoration")
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


def main() -> None:
    args = parse_args()
    cfg, stage_a_model, encoder, decoder, val_loader, device = load_stage_b_decoder(args.config, args.overrides, args.checkpoint)
    perceptual = VGGPerceptualLoss().to(device).eval()
    clean_anchor_t = compute_clean_anchor_t(stage_a_model, encoder, val_loader, device)

    decoder_mode = cfg["model"].get("decoder_mode", "plain")
    input_mode = cfg["model"].get("input_mode", "st")
    totals = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS_proxy": 0.0, "DINO_similarity": 0.0, "num_batches": 0}

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["is_degraded"].to(device)
            deg_mask = y == 1
            if not deg_mask.any():
                continue

            x_deg = x[deg_mask]
            deep = encoder.forward_deep(x_deg)
            shallow = encoder.forward_shallow(x_deg)
            out = stage_a_model(deep)

            if input_mode == "st":
                decoder_in = torch.cat([out["s"], clean_anchor_t.repeat(out["s"].size(0), 1)], dim=-1)
            else:
                decoder_in = stage_a_model.reconstructor(out["s"], clean_anchor_t.repeat(out["s"].size(0), 1))

            if decoder_mode == "dinotok":
                x_restored = decoder(decoder_in, shallow)
            else:
                x_restored = decoder(decoder_in, None)

            deep_restored = encoder.forward_deep(x_restored)
            totals["PSNR"] += float(batch_psnr(x_restored, x_deg))
            totals["SSIM"] += float(batch_ssim_global(x_restored, x_deg))
            totals["LPIPS_proxy"] += float(perceptual(x_restored, x_deg))
            totals["DINO_similarity"] += float(f.cosine_similarity(deep_restored.mean(dim=1), deep.mean(dim=1), dim=-1).mean())
            totals["num_batches"] += 1

    n = max(int(totals["num_batches"]), 1)
    report = {
        "PSNR": totals["PSNR"] / n,
        "SSIM": totals["SSIM"] / n,
        "LPIPS_proxy": totals["LPIPS_proxy"] / n,
        "DINO_similarity": totals["DINO_similarity"] / n,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_metrics_json(out_dir / "restoration_metrics.json", report)
    save_table_csv(str(out_dir / "restoration_metrics.csv"), [report])
    print(report)


if __name__ == "__main__":
    main()
