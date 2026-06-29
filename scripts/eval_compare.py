"""Compare Vanilla A*, Neural A*, and diffusion-based Neural A*."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from neural_astar.planner import DiffusionAstar, NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import set_global_seeds
from omegaconf import OmegaConf


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


def _metrics(outputs, va_outputs) -> dict[str, float]:
    pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    p_opt = (pathlen_astar == pathlen_model).mean()

    exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    exp_model = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    p_exp = np.maximum((exp_astar - exp_model) / exp_astar, 0.0).mean()

    h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))
    return {"p_opt": float(p_opt), "p_exp": float(p_exp), "h_mean": float(h_mean)}


@torch.no_grad()
def _evaluate(name: str, planner, loader, max_batches: int | None, device: torch.device) -> dict[str, float]:
    planner.eval()
    planner.to(device)
    vanilla = VanillaAstar().to(device).eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        map_designs, start_maps, goal_maps, _ = batch
        map_designs = map_designs.to(device)
        start_maps = start_maps.to(device)
        goal_maps = goal_maps.to(device)
        start_maps = start_maps[:, :1]
        goal_maps = goal_maps[:, :1]
        va_outputs = vanilla(map_designs, start_maps, goal_maps)
        outputs = planner(map_designs, start_maps, goal_maps)
        rows.append(_metrics(outputs, va_outputs))

    summary = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("p_opt", "p_exp", "h_mean")
    }
    print(
        f"{name:16s} "
        f"p_opt={summary['p_opt']:.4f} "
        f"p_exp={summary['p_exp']:.4f} "
        f"h_mean={summary['h_mean']:.4f}"
    )
    return summary


@hydra.main(version_base=None, config_path="config", config_name="eval_compare")
def main(config):
    set_global_seeds(config.seed)
    device = torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else ("cpu" if config.device == "auto" else config.device)
    )
    loader = create_dataloader(
        config.dataset + ".npz",
        config.split,
        config.batch_size,
        shuffle=False,
    )

    planners = [("VanillaAstar", VanillaAstar())]

    if config.neural_ckpt:
        neural = NeuralAstar(
            encoder_input=config.neural.encoder_input,
            encoder_arch=config.neural.encoder_arch,
            encoder_depth=config.neural.encoder_depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        _load_planner_state(neural, str(Path(config.neural_ckpt)))
        planners.append(("NeuralAstar", neural))
    else:
        print("NeuralAstar skipped: set neural_ckpt=path/to/checkpoint.ckpt")

    if config.diffusion_ckpt:
        diffusion = DiffusionAstar(
            unet_config=OmegaConf.to_container(config.diffusion_model.unet, resolve=True),
            diffusion_config=OmegaConf.to_container(config.diffusion_model.diffusion, resolve=True),
            Tmax=1.0,
            start_goal_sigma=config.diffusion_model.start_goal_sigma,
        )
        _load_planner_state(diffusion, str(Path(config.diffusion_ckpt)))
        planners.append(("DiffusionAstar", diffusion))
    else:
        print("DiffusionAstar skipped: set diffusion_ckpt=path/to/checkpoint.ckpt")

    for name, planner in planners:
        _evaluate(name, planner, loader, config.max_batches, device)


if __name__ == "__main__":
    main()
