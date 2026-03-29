from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_pca_scatter(points_2d: np.ndarray, labels: np.ndarray, title: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.7, s=12)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
