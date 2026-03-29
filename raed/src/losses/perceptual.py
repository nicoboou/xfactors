from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import VGG16_Weights, vgg16


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES)
        except Exception:
            model = vgg16(weights=None)
        self.features = nn.ModuleList(
            [
                model.features[:4].eval(),
                model.features[4:9].eval(),
                model.features[9:16].eval(),
            ]
        )
        for block in self.features:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=x_hat.device)
        feat_hat, feat = x_hat, x
        for block in self.features:
            feat_hat = block(feat_hat)
            feat = block(feat)
            loss = loss + f.l1_loss(feat_hat, feat)
        return loss
