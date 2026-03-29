from __future__ import annotations

import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VariationalFactorizer(nn.Module):
    def __init__(self, in_dim: int, latent_dim_s: int, latent_dim_t: int, hidden_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.backbone = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.mu_s = nn.Linear(hidden_dim, latent_dim_s)
        self.logvar_s = nn.Linear(hidden_dim, latent_dim_s)
        self.mu_t = nn.Linear(hidden_dim, latent_dim_t)
        self.logvar_t = nn.Linear(hidden_dim, latent_dim_t)

    def forward(self, deep_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = deep_tokens.mean(dim=1)
        h = self.backbone(pooled)
        mu_s, logvar_s = self.mu_s(h), self.logvar_s(h)
        mu_t, logvar_t = self.mu_t(h), self.logvar_t(h)
        s = reparameterize(mu_s, logvar_s)
        t = reparameterize(mu_t, logvar_t)
        return {"pooled": pooled, "mu_s": mu_s, "logvar_s": logvar_s, "mu_t": mu_t, "logvar_t": logvar_t, "s": s, "t": t}
