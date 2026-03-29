from __future__ import annotations

import torch
import torch.nn as nn


class DinoTokFusion(nn.Module):
    def __init__(self, deep_dim: int, shallow_dim: int, out_dim: int):
        super().__init__()
        self.deep_proj = nn.Linear(deep_dim, out_dim)
        self.shallow_proj = nn.Linear(shallow_dim, out_dim)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, deep: torch.Tensor, shallow: torch.Tensor | None) -> torch.Tensor:
        deep_h = self.deep_proj(deep)
        if shallow is None:
            shallow_h = torch.zeros_like(deep_h)
        else:
            shallow_h = self.shallow_proj(shallow)
        return self.fusion(torch.cat([deep_h, shallow_h], dim=-1))
