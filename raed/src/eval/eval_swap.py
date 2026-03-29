from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
from sklearn.linear_model import LogisticRegression

from raed.src.eval.common import load_stage_a, save_table_csv
from raed.src.utils.logging import dump_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage A latent swaps")
    parser.add_argument("--config", type=str, default="raed/configs/stage_a_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="raed/outputs/eval_swap")
    parser.add_argument("--max_pairs", type=int, default=256)
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


@torch.no_grad()
def gather(model, encoder, loader, device):
    s_all, t_all, y_all, pooled_all = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["is_degraded"].to(device)
        deep = encoder.forward_deep(x)
        out = model(deep)
        s_all.append(out["s"].cpu())
        t_all.append(out["t"].cpu())
        pooled_all.append(out["pooled"].cpu())
        y_all.append(y.cpu())
    return torch.cat(s_all), torch.cat(t_all), torch.cat(y_all), torch.cat(pooled_all)


def main() -> None:
    args = parse_args()
    _cfg, model, encoder, val_loader, device = load_stage_a(args.config, args.overrides, args.checkpoint)
    s, t, y, pooled = gather(model, encoder, val_loader, device)

    y_np = y.numpy()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(t.numpy(), y_np)

    idx_clean = torch.where(y == 0)[0]
    idx_deg = torch.where(y == 1)[0]
    n_pairs = min(len(idx_clean), len(idx_deg), args.max_pairs)
    if n_pairs == 0:
        raise RuntimeError("Need both clean and degraded samples for swap evaluation")

    clean_pick = idx_clean[torch.randperm(len(idx_clean))[:n_pairs]]
    deg_pick = idx_deg[torch.randperm(len(idx_deg))[:n_pairs]]

    s_clean, t_clean, pooled_clean = s[clean_pick].to(device), t[clean_pick].to(device), pooled[clean_pick].to(device)
    s_deg, t_deg, pooled_deg = s[deg_pick].to(device), t[deg_pick].to(device), pooled[deg_pick].to(device)

    h_clean_with_deg = model.reconstructor(s_clean, t_deg)
    h_deg_with_clean = model.reconstructor(s_deg, t_clean)

    sem_clean = f.cosine_similarity(h_clean_with_deg, pooled_clean, dim=-1).mean().item()
    sem_deg = f.cosine_similarity(h_deg_with_clean, pooled_deg, dim=-1).mean().item()

    pred_clean_with_deg = clf.predict_proba(t_deg.cpu().numpy())[:, 1].mean().item()
    pred_deg_with_clean = clf.predict_proba(t_clean.cpu().numpy())[:, 1].mean().item()

    report = {
        "semantic_cosine_clean_src": sem_clean,
        "semantic_cosine_deg_src": sem_deg,
        "degraded_prob_clean_with_deg_t": pred_clean_with_deg,
        "degraded_prob_deg_with_clean_t": pred_deg_with_clean,
        "num_pairs": int(n_pairs),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_metrics_json(out_dir / "swap_metrics.json", report)
    save_table_csv(str(out_dir / "swap_metrics.csv"), [report])
    print(report)


if __name__ == "__main__":
    main()
