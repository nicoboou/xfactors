from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from raed.src.eval.common import collect_latents, load_stage_a, save_table_csv
from raed.src.utils.logging import dump_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage A probes")
    parser.add_argument("--config", type=str, default="raed/configs/stage_a_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="raed/outputs/eval_probes")
    parser.add_argument("overrides", nargs="*", default=[])
    return parser.parse_args()


def run_probe(features: np.ndarray, labels: np.ndarray, seed: int = 2026) -> dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=seed, stratify=labels)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)[:, 1]
    return {"accuracy": float(accuracy_score(y_test, pred)), "auroc": float(roc_auc_score(y_test, prob))}


def main() -> None:
    args = parse_args()
    cfg, model, encoder, val_loader, _device = load_stage_a(args.config, args.overrides, args.checkpoint)
    s, t, y = collect_latents(model, encoder, val_loader, _device)

    s_metrics = run_probe(s, y, seed=int(cfg["seed"]))
    t_metrics = run_probe(t, y, seed=int(cfg["seed"]))

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"probe_s": s_metrics, "probe_t": t_metrics}
    dump_metrics_json(output_dir / "probe_metrics.json", report)

    rows = [
        {"latent": "s", **s_metrics},
        {"latent": "t", **t_metrics},
    ]
    save_table_csv(str(output_dir / "probe_metrics.csv"), rows)
    print(report)


if __name__ == "__main__":
    main()
