"""Diffusion-based Neural A* search.

Stage 1 (this module's encoder): a conditional DDPM U-Net generates a
"bright path" trajectory heatmap from (obstacle, start, goal) conditions, which
is flipped into a cost map (cost = 1 - heatmap).

Stage 2: the exact same DifferentiableAstar / pq_astar as NeuralAstar, so the
two methods differ only in how the cost map is produced.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.heatmap import mask_to_gaussian_heatmap
from .astar import VanillaAstar
from .diffusion import DiffusionScheduler
from .diffusion_unet import build_unet_from_config
from .differentiable_astar import AstarOutput, DifferentiableAstar


class DiffusionAstar(VanillaAstar):
    def __init__(
        self,
        unet_config: dict | None = None,
        diffusion_config: dict | None = None,
        g_ratio: float = 0.5,
        Tmax: float = 1.0,
        start_goal_sigma: float = 1.0,
        use_differentiable_astar: bool = True,
    ):
        """Diffusion Neural A* search.

        Args:
            unet_config: dict passed to build_unet_from_config. Defaults to a
                1-channel-target baseline U-Net with 3 condition channels.
            diffusion_config: dict passed to DiffusionScheduler.
            g_ratio: ratio between g(v) + h(v) in A*. Defaults to 0.5.
            Tmax: exploration budget for differentiable A*. Defaults to 1.0.
            start_goal_sigma: Gaussian sigma (pixels) for start/goal condition
                heatmaps. Defaults to 1.0 (tuned for 32x32 mazes).
            use_differentiable_astar: if False, use pq_astar at inference.
        """
        nn.Module.__init__(self)
        unet_config = dict(unet_config or {})
        unet_config.setdefault("model_type", "baseline")
        unet_config.setdefault("target_channels", 1)
        diffusion_config = dict(diffusion_config or {"num_steps": 100, "beta_schedule": "cosine"})

        self.unet = build_unet_from_config(unet_config)
        self.diffusion = DiffusionScheduler(**diffusion_config)
        self.astar = DifferentiableAstar(g_ratio=g_ratio, Tmax=Tmax)

        self.unet_config = unet_config
        self.start_goal_sigma = float(start_goal_sigma)
        self.g_ratio = g_ratio
        self.Tmax = Tmax
        self.use_differentiable_astar = use_differentiable_astar

    def build_condition(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Build the 3 DDPM condition channels (obstacle, start_hm, goal_hm).

        map_designs is passed through as the obstacle condition (1 = passable,
        same polarity used by the A* obstacle mask). start/goal one-hot masks
        are blurred into Gaussian heatmaps.
        """
        obstacle = map_designs
        start_hm = mask_to_gaussian_heatmap(start_maps, sigma=self.start_goal_sigma)
        goal_hm = mask_to_gaussian_heatmap(goal_maps, sigma=self.start_goal_sigma)
        return obstacle, start_hm, goal_hm

    def encode(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
    ) -> torch.tensor:
        """Sample a trajectory heatmap via DDPM and flip it into a cost map."""
        obstacle, start_hm, goal_hm = self.build_condition(map_designs, start_maps, goal_maps)
        target_channels = int(self.unet_config.get("target_channels", 1))
        shape = (
            map_designs.shape[0],
            target_channels,
            map_designs.shape[-2],
            map_designs.shape[-1],
        )
        heatmap = self.diffusion.sample(self.unet, obstacle, start_hm, goal_hm, shape=shape)
        heatmap = heatmap[:, :1]  # keep the path-occupancy channel
        cost_maps = 1.0 - heatmap.clamp(0.0, 1.0)
        return cost_maps

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """Perform diffusion Neural A* search (inference / evaluation only)."""
        cost_maps = self.encode(map_designs, start_maps, goal_maps)
        obstacles_maps = map_designs

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )
