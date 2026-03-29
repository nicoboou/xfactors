from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as f


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps: int, schedule: str = "linear"):
        super().__init__()
        self.num_steps = int(num_steps)
        if schedule == "linear":
            betas = torch.linspace(1e-4, 2e-2, steps=self.num_steps)
        else:
            s = 0.008
            x = torch.linspace(0, self.num_steps, self.num_steps + 1)
            acp = torch.cos(((x / self.num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            acp = acp / acp[0]
            betas = 1 - (acp[1:] / acp[:-1])
            betas = betas.clamp(1e-5, 0.999)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
        return a * x0 + b * noise, noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (xt.ndim - 1)))
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (xt.ndim - 1)))
        return (xt - b * eps) / a.clamp_min(1e-8)


class SemanticConditionedDenoiser(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.s_proj = nn.Linear(s_dim, hidden_dim)
        self.down1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.up1 = nn.ConvTranspose2d(hidden_dim, 64, kernel_size=4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x_t.shape
        h1 = f.gelu(self.down1(x_t))
        h2 = f.gelu(self.down2(h1))
        hb, hc, hh, hw = h2.shape
        seq = h2.flatten(2).transpose(1, 2)
        time_emb = self.time_mlp(timestep_embedding(t, hc)).unsqueeze(1)
        seq = seq + time_emb
        s_token = self.s_proj(s).unsqueeze(1)
        seq = self.attn(query=seq, key=s_token, value=s_token)[0]
        h2 = seq.transpose(1, 2).reshape(hb, hc, hh, hw)
        h_up = f.gelu(self.up1(h2))
        if h_up.shape[-2:] != (h, w):
            h_up = f.interpolate(h_up, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(h_up)


class LatentDenoiser(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.in_proj = nn.Linear(z_dim, hidden_dim)
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)))
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z_t)
        h = h + self.time_proj(timestep_embedding(t, h.shape[-1]))
        for block in self.blocks:
            h = h + block(h)
        return self.out_proj(h)


class SemanticConditionedPixelDiffusion(nn.Module):
    def __init__(self, s_dim: int, num_steps: int, schedule: str = "linear", hidden_dim: int = 256):
        super().__init__()
        self.schedule = DiffusionSchedule(num_steps=num_steps, schedule=schedule)
        self.denoiser = SemanticConditionedDenoiser(s_dim=s_dim, hidden_dim=hidden_dim)

    def loss(self, x0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.schedule.num_steps, (b,), device=x0.device)
        xt, noise = self.schedule.q_sample(x0, t)
        pred = self.denoiser(xt, t, s)
        return f.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, s: torch.Tensor, image_size: int, steps: int | None = None) -> torch.Tensor:
        b = s.size(0)
        x = torch.randn(b, 3, image_size, image_size, device=s.device)
        n_steps = self.schedule.num_steps if steps is None else min(int(steps), self.schedule.num_steps)
        for i in reversed(range(n_steps)):
            t = torch.full((b,), i, device=s.device, dtype=torch.long)
            eps = self.denoiser(x, t, s)
            x0 = self.schedule.predict_x0_from_eps(x, t, eps)
            if i > 0:
                z = torch.randn_like(x)
                alpha = self.schedule.alphas[i]
                alpha_bar = self.schedule.alphas_cumprod[i]
                beta = self.schedule.betas[i]
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) + torch.sqrt(beta) * z
            else:
                x = x0
        return x.clamp(0.0, 1.0)


class LatentRAEDDiffusion(nn.Module):
    def __init__(self, z_dim: int, num_steps: int, schedule: str = "linear", hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        self.schedule = DiffusionSchedule(num_steps=num_steps, schedule=schedule)
        self.denoiser = LatentDenoiser(z_dim=z_dim, hidden_dim=hidden_dim, depth=depth)

    def loss(self, z0: torch.Tensor) -> torch.Tensor:
        b = z0.size(0)
        t = torch.randint(0, self.schedule.num_steps, (b,), device=z0.device)
        zt, noise = self.schedule.q_sample(z0, t)
        pred = self.denoiser(zt, t)
        return f.mse_loss(pred, noise)

    @torch.no_grad()
    def refine(self, z_init: torch.Tensor, refine_steps: int = 20) -> torch.Tensor:
        b = z_init.size(0)
        steps = max(1, min(int(refine_steps), self.schedule.num_steps - 1))
        t_start = torch.full((b,), steps, device=z_init.device, dtype=torch.long)
        zt, _ = self.schedule.q_sample(z_init, t_start)
        for i in reversed(range(steps)):
            t = torch.full((b,), i, device=z_init.device, dtype=torch.long)
            eps = self.denoiser(zt, t)
            x0 = self.schedule.predict_x0_from_eps(zt, t, eps)
            if i > 0:
                noise = torch.randn_like(zt)
                alpha = self.schedule.alphas[i]
                alpha_bar = self.schedule.alphas_cumprod[i]
                beta = self.schedule.betas[i]
                zt = (1 / torch.sqrt(alpha)) * (zt - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) + torch.sqrt(beta) * noise
            else:
                zt = x0
        return zt
