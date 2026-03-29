from __future__ import annotations

import torch
import torch.nn.functional as f


def deep_reconstruction_loss(h_hat: torch.Tensor, h_target: torch.Tensor, lambda_mse: float, lambda_cos: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = f.mse_loss(h_hat, h_target)
    cosine = 1.0 - f.cosine_similarity(h_hat, h_target, dim=-1).mean()
    total = lambda_mse * mse + lambda_cos * cosine
    return total, mse, cosine
