"""Gaussian heatmap helpers for diffusion-based planning.

neural-astar stores start/goal/path as binary grid masks (not continuous xy
points), so these helpers turn a grid mask into a smooth "bright path" heatmap
via a fixed Gaussian blur, then per-sample normalize to [0, 1]. For a 1-pixel
wide continuous grid path this is equivalent in spirit to GDPP's per-point
max-of-gaussians (gdpp/src/gdpp/core/heatmap.py), but fully vectorized over a
batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_kernel2d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = torch.outer(g1d, g1d)
    return kernel / kernel.max()  # peak 1.0, so a single point maps to a unit-peak blob


def mask_to_gaussian_heatmap(mask: torch.Tensor, sigma: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Blur a binary grid mask into a [0, 1] heatmap (bright where mask==1).

    Args:
        mask: [B, 1, H, W] (or [B, H, W]) binary tensor.
        sigma: Gaussian sigma in pixels.

    Returns:
        [B, 1, H, W] heatmap, each sample normalized so its max is 1
        (all-zero samples stay all-zero).
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    x = mask.float()
    k = _gaussian_kernel2d(sigma, x.device, x.dtype)
    k = k.reshape(1, 1, *k.shape)
    pad = k.shape[-1] // 2
    blurred = F.conv2d(x, k, padding=pad)
    peak = blurred.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (blurred / peak).clamp(0.0, 1.0)
