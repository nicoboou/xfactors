from __future__ import annotations

import torch
import torch.nn.functional as f


def supervised_info_nce(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [B,D], got {tuple(embeddings.shape)}")
    z = f.normalize(embeddings, dim=1)
    logits = z @ z.t() / temperature
    logits = logits - torch.diag_embed(torch.full((logits.size(0),), 1e9, device=logits.device))
    labels = labels.view(-1, 1)
    positives = labels.eq(labels.t()).float()
    positives.fill_diagonal_(0.0)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    denom = positives.sum(dim=1).clamp_min(1.0)
    loss = -(positives * log_probs).sum(dim=1) / denom
    return loss.mean()
