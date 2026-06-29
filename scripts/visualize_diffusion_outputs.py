"""Visualize planner outputs for Neural A* / diffusion / GDPP-no-MSE.

This script samples start-goal problems from the neural-astar maze dataset and
writes per-scene panels showing the map, paths, search histories, generated
cost maps, and diffusion heatmaps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_astar.planner import DiffusionAstar, NeuralAstar, VanillaAstar
from neural_astar.planner.gdpp_no_mse_astar import (
    _astar_on_heatmap_with_history,
    _heatmap_to_cost_map,
    _mask_to_xy,
)
from neural_astar.utils.data import MazeDataset
from neural_astar.utils.training import set_global_seeds
from neural_astar.planner.differentiable_astar import AstarOutput


def _load_planner_state(planner, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    has_planner_prefix = any(key.startswith("planner.") for key in state_dict)
    extracted = {
        key[len("planner.") :] if has_planner_prefix else key: value
        for key, value in state_dict.items()
        if not has_planner_prefix or key.startswith("planner.")
    }
    missing, unexpected = planner.load_state_dict(extracted, strict=False)
    if missing:
        print(f"missing keys for {checkpoint_path}: {missing}")
    if unexpected:
        print(f"unexpected keys for {checkpoint_path}: {unexpected}")


def _normalize(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if mask is not None:
        finite = finite & mask
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[finite], [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _path_length(path_map: np.ndarray) -> float:
    return float(np.asarray(path_map).sum())


def _expanded(history_map: np.ndarray) -> float:
    return float(np.asarray(history_map).sum())


def _p_exp(exp_ref: float, exp_model: float) -> float:
    if exp_ref <= 0:
        return 0.0
    return float(max((exp_ref - exp_model) / exp_ref, 0.0))


def _overlay(ax, map_design, start, goal, title, path=None, history=None, gt=None):
    height, width = map_design.shape
    rgb = np.ones((height, width, 3), dtype=np.float32) * 0.96
    rgb[map_design < 0.5] = np.array([0.05, 0.05, 0.05])

    if history is not None:
        rgb[history > 0.5] = 0.55 * rgb[history > 0.5] + 0.45 * np.array([0.1, 0.75, 0.2])
    if gt is not None:
        rgb[gt > 0.5] = np.array([1.0, 0.68, 0.15])
    if path is not None:
        rgb[path > 0.5] = np.array([0.95, 0.05, 0.05])
    rgb[start > 0.5] = np.array([0.05, 0.35, 1.0])
    rgb[goal > 0.5] = np.array([0.75, 0.0, 0.95])

    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _heat(ax, data, title, cmap="magma", blocked=None):
    shown = np.asarray(data, dtype=np.float32)
    valid = None if blocked is None else ~blocked
    shown = _normalize(shown, mask=valid)
    if blocked is not None:
        shown = np.ma.masked_where(blocked, shown)
    ax.imshow(shown, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_panel(
    output_file: Path,
    map_design: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    opt_traj: np.ndarray,
    vanilla: AstarOutput,
    neural: AstarOutput | None,
    neural_cost: np.ndarray | None,
    diffusion: AstarOutput | None,
    diffusion_heatmap: np.ndarray | None,
    diffusion_cost: np.ndarray | None,
    gdpp: AstarOutput | None,
    gdpp_cost: np.ndarray | None,
    title: str,
) -> None:
    blocked = map_design < 0.5
    fig, axes = plt.subplots(3, 4, figsize=(11, 8), constrained_layout=True)
    axes = axes.flatten()

    _overlay(axes[0], map_design, start, goal, "map + GT opt_traj", gt=opt_traj)
    _overlay(
        axes[1],
        map_design,
        start,
        goal,
        "Vanilla path/history",
        path=_as_numpy(vanilla.paths[0, 0]),
        history=_as_numpy(vanilla.histories[0, 0]),
    )

    if neural is not None and neural_cost is not None:
        _heat(axes[2], neural_cost, "Neural cost", cmap="viridis", blocked=blocked)
        _overlay(
            axes[3],
            map_design,
            start,
            goal,
            "Neural path/history",
            path=_as_numpy(neural.paths[0, 0]),
            history=_as_numpy(neural.histories[0, 0]),
        )
    else:
        axes[2].axis("off")
        axes[3].axis("off")

    if diffusion is not None and diffusion_heatmap is not None and diffusion_cost is not None:
        _heat(axes[4], diffusion_heatmap, "Diffusion heatmap", cmap="magma", blocked=blocked)
        _heat(axes[5], diffusion_cost, "Diffusion cost = 1 - heatmap", cmap="viridis", blocked=blocked)
        _overlay(
            axes[6],
            map_design,
            start,
            goal,
            "DiffusionAstar path/history",
            path=_as_numpy(diffusion.paths[0, 0]),
            history=_as_numpy(diffusion.histories[0, 0]),
        )
    else:
        axes[4].axis("off")
        axes[5].axis("off")
        axes[6].axis("off")

    if gdpp is not None and gdpp_cost is not None:
        _heat(axes[7], gdpp_cost, "GDPP -log heatmap cost", cmap="viridis", blocked=blocked)
        _overlay(
            axes[8],
            map_design,
            start,
            goal,
            "GDPPNoMSE path/history",
            path=_as_numpy(gdpp.paths[0, 0]),
            history=_as_numpy(gdpp.histories[0, 0]),
        )
    else:
        axes[7].axis("off")
        axes[8].axis("off")

    axes[9].axis("off")
    axes[10].axis("off")
    axes[11].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def _make_contact_sheet(sample_files: list[Path], output_file: Path, max_images: int) -> None:
    if not sample_files:
        return
    import math

    files = sample_files[:max_images]
    cols = min(4, len(files))
    rows = int(math.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, file in zip(axes, files):
        ax.imshow(plt.imread(file))
        ax.set_title(file.stem, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(files):]:
        ax.axis("off")
    fig.savefig(output_file, dpi=130)
    plt.close(fig)


def _build_planners(args, device: torch.device):
    vanilla = VanillaAstar().to(device).eval()

    neural = None
    if args.neural_ckpt:
        neural = NeuralAstar(
            encoder_input=args.neural_encoder_input,
            encoder_arch=args.neural_encoder_arch,
            encoder_depth=args.neural_encoder_depth,
            learn_obstacles=False,
            Tmax=1.0,
        ).to(device).eval()
        _load_planner_state(neural, args.neural_ckpt)

    diffusion = None
    if args.diffusion_ckpt:
        unet_config = {
            "model_type": args.diffusion_model_type,
            "target_channels": 1,
            "base_channels": args.base_channels,
            "time_emb_dim": args.time_emb_dim,
            "channel_mults": tuple(args.channel_mults),
        }
        diffusion_config = {
            "num_steps": args.diffusion_steps,
            "beta_schedule": args.beta_schedule,
            "clip_sample": True,
        }
        diffusion = DiffusionAstar(
            unet_config=unet_config,
            diffusion_config=diffusion_config,
            Tmax=1.0,
            start_goal_sigma=args.start_goal_sigma,
        ).to(device).eval()
        _load_planner_state(diffusion, args.diffusion_ckpt)

    return vanilla, neural, diffusion


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="planning-datasets/data/mpd/mazes_032_moore_c8.npz")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--neural-ckpt")
    parser.add_argument("--diffusion-ckpt")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-schedule", default="cosine")
    parser.add_argument("--diffusion-model-type", default="baseline")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-emb-dim", type=int, default=256)
    parser.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--start-goal-sigma", type=float, default=1.0)
    parser.add_argument("--gdpp-lam", type=float, default=1.0)
    parser.add_argument("--gdpp-heuristic-weight", type=float, default=1.0)
    parser.add_argument("--neural-encoder-input", default="m+")
    parser.add_argument("--neural-encoder-arch", default="CNN")
    parser.add_argument("--neural-encoder-depth", type=int, default=4)
    parser.add_argument("--contact-sheet-max", type=int, default=32)
    args = parser.parse_args()

    set_global_seeds(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = MazeDataset(args.dataset, args.split, num_starts=1)
    vanilla, neural, diffusion = _build_planners(args, device)

    rows = []
    sample_files = []
    for sample_idx in range(args.num_samples):
        map_design_np, start_np, goal_np, opt_np = dataset[sample_idx % len(dataset)]
        map_design = torch.from_numpy(map_design_np).unsqueeze(0).to(device)
        start = torch.from_numpy(start_np[:1]).unsqueeze(0).to(device)
        goal = torch.from_numpy(goal_np[:1]).unsqueeze(0).to(device)
        opt_traj = opt_np[:1]

        vanilla_out = vanilla(map_design, start, goal)

        neural_out = None
        neural_cost_np = None
        if neural is not None:
            neural_cost = neural.encode(map_design, start, goal)
            neural_out = neural.perform_astar(neural_cost, start, goal, map_design)
            neural_cost_np = _as_numpy(neural_cost[0, 0])

        diffusion_out = None
        diffusion_heatmap_np = None
        diffusion_cost_np = None
        gdpp_out = None
        gdpp_cost_np = None
        if diffusion is not None:
            heatmap = diffusion.sample_heatmap(map_design, start, goal).clamp(0.0, 1.0)
            diffusion_cost = 1.0 - heatmap
            diffusion_out = diffusion.perform_astar(diffusion_cost, start, goal, map_design)

            diffusion_heatmap_np = _as_numpy(heatmap[0, 0])
            diffusion_cost_np = _as_numpy(diffusion_cost[0, 0])
            map_np = map_design_np[0]
            start_xy = _mask_to_xy(start_np[0])
            goal_xy = _mask_to_xy(goal_np[0])
            blocked_map = 1.0 - map_np
            gdpp_history, gdpp_path = _astar_on_heatmap_with_history(
                diffusion_heatmap_np,
                blocked_map,
                start_xy,
                goal_xy,
                lam=args.gdpp_lam,
                heuristic_weight=args.gdpp_heuristic_weight,
            )
            gdpp_out = AstarOutput(
                torch.from_numpy(gdpp_history).reshape(1, 1, *gdpp_history.shape).to(device),
                torch.from_numpy(gdpp_path).reshape(1, 1, *gdpp_path.shape).to(device),
            )
            gdpp_cost_np = _heatmap_to_cost_map(diffusion_heatmap_np, blocked_map)

        ref_len = _path_length(_as_numpy(vanilla_out.paths[0, 0]))
        ref_exp = _expanded(_as_numpy(vanilla_out.histories[0, 0]))
        method_outputs = {
            "VanillaAstar": vanilla_out,
            "NeuralAstar": neural_out,
            "DiffusionAstar": diffusion_out,
            "GDPPNoMSE": gdpp_out,
        }
        for method, output in method_outputs.items():
            if output is None:
                continue
            path_len = _path_length(_as_numpy(output.paths[0, 0]))
            expanded = _expanded(_as_numpy(output.histories[0, 0]))
            rows.append(
                {
                    "sample": sample_idx,
                    "dataset_index": sample_idx % len(dataset),
                    "method": method,
                    "path_len": path_len,
                    "expanded": expanded,
                    "p_opt": float(path_len == ref_len),
                    "p_exp": _p_exp(ref_exp, expanded),
                }
            )

        sample_file = sample_dir / f"sample_{sample_idx:04d}.png"
        _save_panel(
            sample_file,
            map_design_np[0],
            start_np[0],
            goal_np[0],
            opt_traj[0],
            vanilla_out,
            neural_out,
            neural_cost_np,
            diffusion_out,
            diffusion_heatmap_np,
            diffusion_cost_np,
            gdpp_out,
            gdpp_cost_np,
            title=f"sample={sample_idx} dataset_index={sample_idx % len(dataset)}",
        )
        sample_files.append(sample_file)
        if (sample_idx + 1) % 25 == 0:
            print(f"wrote {sample_idx + 1}/{args.num_samples} samples")

    metrics_file = output_dir / "per_sample_metrics.csv"
    with metrics_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "dataset_index", "method", "path_len", "expanded", "p_opt", "p_exp"],
        )
        writer.writeheader()
        writer.writerows(rows)

    _make_contact_sheet(sample_files, output_dir / "contact_sheet.png", args.contact_sheet_max)
    print(f"wrote samples to {sample_dir}")
    print(f"wrote metrics to {metrics_file}")
    print(f"wrote contact sheet to {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
