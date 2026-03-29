from __future__ import annotations

import torch
import torch.nn as nn


class FrozenDinoEncoder(nn.Module):
    def __init__(self, backbone: str, repo: str = "facebookresearch/dinov2", use_cls_token: bool = False, freeze: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.repo = repo
        self.use_cls_token = use_cls_token
        self.model = torch.hub.load(repo, backbone)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.model.forward_features(x)
        if "x_norm_patchtokens" not in out:
            raise RuntimeError("DINO output missing x_norm_patchtokens")
        deep_tokens = out["x_norm_patchtokens"]
        cls_token = out.get("x_norm_clstoken")
        if self.use_cls_token and cls_token is not None:
            deep_tokens = torch.cat([cls_token.unsqueeze(1), deep_tokens], dim=1)
        return deep_tokens, cls_token

    @torch.no_grad()
    def forward_deep(self, x: torch.Tensor) -> torch.Tensor:
        deep_tokens, _ = self._extract(x)
        return deep_tokens

    @torch.no_grad()
    def forward_shallow(self, x: torch.Tensor) -> torch.Tensor | None:
        _deep_tokens, cls_token = self._extract(x)
        return cls_token

    @torch.no_grad()
    def forward_all(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        deep_tokens, cls_token = self._extract(x)
        return {"deep_tokens": deep_tokens, "cls_token": cls_token, "shallow_tokens": cls_token}
