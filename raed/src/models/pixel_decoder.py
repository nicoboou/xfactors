from __future__ import annotations

import torch
import torch.nn as nn

from .dinotok_fusion import DinoTokFusion


class _DecoderBackbone(nn.Module):
    def __init__(self, in_dim: int, base_channels: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_dim, base_channels * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), -1, 7, 7)
        return self.decoder(x)


class PlainPixelDecoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.backbone = _DecoderBackbone(in_dim=in_dim)

    def forward(self, z: torch.Tensor, shallow: torch.Tensor | None = None) -> torch.Tensor:
        _ = shallow
        return self.backbone(z)


class DinoTokPixelDecoder(nn.Module):
    def __init__(self, deep_dim: int, shallow_dim: int, fused_dim: int):
        super().__init__()
        self.fusion = DinoTokFusion(deep_dim=deep_dim, shallow_dim=shallow_dim, out_dim=fused_dim)
        self.backbone = _DecoderBackbone(in_dim=fused_dim)

    def forward(self, deep: torch.Tensor, shallow: torch.Tensor | None) -> torch.Tensor:
        fused = self.fusion(deep, shallow)
        return self.backbone(fused)
