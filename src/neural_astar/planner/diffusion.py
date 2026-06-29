"""Conditional DDPM scheduler for diffusion-based cost map generation.

Ported from GDPP (gdpp/src/gdpp/core/diffusion.py). Depends only on torch.
"""

from __future__ import annotations

import math

import torch


def _cosine_beta_schedule(num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        alpha_bar_t1 = math.cos((t1 + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar_t2 = math.cos((t2 + 0.008) / 1.008 * math.pi / 2) ** 2
        betas.append(min(1.0 - alpha_bar_t2 / alpha_bar_t1, max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = values.to(device=timesteps.device).gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


class DiffusionScheduler:
    def __init__(
        self,
        num_steps: int = 100,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        clip_sample: bool = True,
    ) -> None:
        self.num_steps = int(num_steps)
        self.clip_sample = bool(clip_sample)
        if beta_schedule in {"cosine", "squaredcos_cap_v2"}:
            betas = _cosine_beta_schedule(self.num_steps)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def to(self, device: torch.device | str) -> "DiffusionScheduler":
        for name in (
            "betas",
            "alphas",
            "alpha_bars",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = extract(self.sqrt_alpha_bars, timesteps, x0.shape)
        sqrt_om = extract(self.sqrt_one_minus_alpha_bars, timesteps, x0.shape)
        return sqrt_ab * x0 + sqrt_om * noise

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        obstacle_map: torch.Tensor,
        start_heatmap: torch.Tensor,
        goal_heatmap: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        betas_t = extract(self.betas, timesteps, x_t.shape)
        sqrt_one_minus_ab = extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t.shape)
        sqrt_recip_alpha = torch.rsqrt(extract(self.alphas, timesteps, x_t.shape))

        pred_noise = model(x_t, obstacle_map, start_heatmap, goal_heatmap, timesteps)
        mean = sqrt_recip_alpha * (x_t - betas_t * pred_noise / sqrt_one_minus_ab)

        noise = torch.randn_like(x_t)
        nonzero_mask = (timesteps != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.dim() - 1)))
        return mean + nonzero_mask * torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        obstacle_map: torch.Tensor,
        start_heatmap: torch.Tensor,
        goal_heatmap: torch.Tensor,
        shape: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        device = obstacle_map.device
        self.to(device)
        if shape is None:
            shape = (obstacle_map.shape[0], 1, obstacle_map.shape[-2], obstacle_map.shape[-1])
        x_t = torch.randn(shape, device=device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, obstacle_map, start_heatmap, goal_heatmap, t)
        return x_t.clamp(0.0, 1.0) if self.clip_sample else x_t
