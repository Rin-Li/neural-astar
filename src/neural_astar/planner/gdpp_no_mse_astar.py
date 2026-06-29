"""GDPP-style diffusion A* without render-MSE refinement.

This planner matches the first half of GDPP's decoder:

    diffusion bright-path heatmap -> GDPP heatmap cost map -> plain 8-neighbor A*

It intentionally does not use neural-astar's DifferentiableAstar and does not
run GDPP's second-stage render-MSE waypoint optimization.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

import numpy as np
import torch

from .diffusion_astar import DiffusionAstar
from .differentiable_astar import AstarOutput


def _heatmap_to_cost_map(
    heatmap: np.ndarray,
    blocked_map: np.ndarray,
    eps: float = 1e-6,
    obstacle_penalty: float = 1e6,
) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    blocked_map = np.asarray(blocked_map, dtype=np.float32)
    h_min, h_max = float(heatmap.min()), float(heatmap.max())
    if h_max > h_min:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return -np.log(heatmap + eps) + obstacle_penalty * (blocked_map > 0.5)


def _mask_to_xy(mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(mask > 0.5)
    if len(rows) == 0:
        raise ValueError("Expected a non-empty one-hot mask")
    return int(cols[0]), int(rows[0])


def _round_xy(point: Iterable[float], height: int, width: int) -> tuple[int, int]:
    x, y = point
    return (
        int(np.clip(round(float(x)), 0, width - 1)),
        int(np.clip(round(float(y)), 0, height - 1)),
    )


def _astar_on_heatmap_with_history(
    heatmap: np.ndarray,
    blocked_map: np.ndarray,
    start_xy: Iterable[float],
    goal_xy: Iterable[float],
    lam: float = 1.0,
    heuristic_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """GDPP A* over an 8-neighbor heatmap grid.

    Returns:
        history_map, path_map with shape [H, W].
    """
    obstacle = np.asarray(blocked_map) > 0.5
    cost_map = _heatmap_to_cost_map(heatmap, obstacle.astype(np.float32))
    height, width = cost_map.shape
    start = _round_xy(start_xy, height, width)
    goal = _round_xy(goal_xy, height, width)

    history_map = np.zeros((height, width), dtype=np.float32)
    path_map = np.zeros((height, width), dtype=np.float32)
    if obstacle[start[1], start[0]] or obstacle[goal[1], goal[0]]:
        return history_map, path_map

    neighbors = [
        (-1, -1, np.sqrt(2.0)), (0, -1, 1.0), (1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),                         (1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),  (0, 1, 1.0),  (1, 1, np.sqrt(2.0)),
    ]

    def heuristic(node: tuple[int, int]) -> float:
        return heuristic_weight * float(np.hypot(node[0] - goal[0], node[1] - goal[1]))

    frontier: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        history_map[current[1], current[0]] = 1.0
        if current == goal:
            break

        cx, cy = current
        for dx, dy, dist in neighbors:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or obstacle[ny, nx]:
                continue
            if dx != 0 and dy != 0 and (obstacle[cy, nx] or obstacle[ny, cx]):
                continue
            step_cost = float(cost_map[ny, nx]) + lam * float(dist)
            tentative = g_score[current] + step_cost
            nxt = (nx, ny)
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                heapq.heappush(frontier, (tentative + heuristic(nxt), nxt))

    if goal not in came_from:
        return history_map, path_map

    node: tuple[int, int] | None = goal
    while node is not None:
        path_map[node[1], node[0]] = 1.0
        node = came_from[node]
    return history_map, path_map


class GDPPNoMSEAstar(DiffusionAstar):
    """GDPP diffusion heatmap + plain A*, without render-MSE refinement."""

    def __init__(
        self,
        *args,
        lam: float = 1.0,
        heuristic_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, use_differentiable_astar=False, **kwargs)
        self.lam = float(lam)
        self.heuristic_weight = float(heuristic_weight)

    @torch.no_grad()
    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        if store_intermediate_results:
            raise ValueError("GDPPNoMSEAstar does not store intermediate A* states")

        heatmaps = self.sample_heatmap(map_designs, start_maps, goal_maps)
        heatmaps_np = heatmaps[:, 0].detach().cpu().numpy()
        map_np = map_designs[:, 0].detach().cpu().numpy()
        starts_np = start_maps[:, 0].detach().cpu().numpy()
        goals_np = goal_maps[:, 0].detach().cpu().numpy()

        histories = np.zeros_like(heatmaps_np, dtype=np.float32)
        paths = np.zeros_like(heatmaps_np, dtype=np.float32)
        for idx in range(len(heatmaps_np)):
            start_xy = _mask_to_xy(starts_np[idx])
            goal_xy = _mask_to_xy(goals_np[idx])
            blocked_map = 1.0 - map_np[idx]
            histories[idx], paths[idx] = _astar_on_heatmap_with_history(
                heatmaps_np[idx],
                blocked_map,
                start_xy,
                goal_xy,
                lam=self.lam,
                heuristic_weight=self.heuristic_weight,
            )

        histories_t = torch.from_numpy(histories).unsqueeze(1).to(map_designs.device)
        paths_t = torch.from_numpy(paths).unsqueeze(1).to(map_designs.device)
        return AstarOutput(histories_t, paths_t)
