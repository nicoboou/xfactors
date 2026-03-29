from __future__ import annotations

import torch
import torch.nn as nn


class DinoReconstructor(nn.Module):
    def __init__(self, latent_dim_s: int, latent_dim_t: int, out_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim_s + latent_dim_t, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, t], dim=-1))
