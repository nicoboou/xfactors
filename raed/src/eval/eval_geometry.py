from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.decomposition import PCA

from raed.src.eval.common import collect_latents, load_stage_a
from raed.src.utils.viz import save_pca_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage A latent geometry")
    parser.add_argument("--config", type=str, default="raed/configs/stage_a_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="raed/outputs/eval_geometry")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _cfg, model, encoder, val_loader, device = load_stage_a(args.config, args.overrides, args.checkpoint)
    s, t, y = collect_latents(model, encoder, val_loader, device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_s = PCA(n_components=2).fit_transform(s)
    pca_t = PCA(n_components=2).fit_transform(t)
    save_pca_scatter(pca_s, y, "PCA of s colored by is_degraded", str(out_dir / "pca_s.png"))
    save_pca_scatter(pca_t, y, "PCA of t colored by is_degraded", str(out_dir / "pca_t.png"))
    print({"pca_s": str(out_dir / "pca_s.png"), "pca_t": str(out_dir / "pca_t.png")})


if __name__ == "__main__":
    main()
