"""Plot diffusion training curves and evaluation metrics.

Examples:
    python scripts/plot_diffusion_results.py \
      --event-dir model/gdpp_diffusion_from_scratch/mazes_032_moore_c8_diffusion/lightning_logs/version_0 \
      --eval-log outputs/gdpp_diffusion_from_scratch/20260630_120000/eval.log \
      --output-dir outputs/gdpp_diffusion_from_scratch/20260630_120000/plots
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_scalars(event_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        raise RuntimeError(
            "TensorBoard is required to read Lightning event files. "
            "Install tensorboard or run this script in an environment that has it."
        ) from exc

    event_files = []
    if event_dir.is_file():
        event_files = [event_dir]
    else:
        event_files = sorted(event_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {event_dir}")

    scalars: dict[str, list[tuple[int, float]]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            scalars.setdefault(tag, [])
            scalars[tag].extend((event.step, float(event.value)) for event in accumulator.Scalars(tag))

    for tag, values in scalars.items():
        scalars[tag] = sorted(values, key=lambda item: item[0])
    return scalars


def _write_scalars_csv(scalars: dict[str, list[tuple[int, float]]], output_file: Path) -> None:
    with output_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "step", "value"])
        for tag, values in sorted(scalars.items()):
            for step, value in values:
                writer.writerow([tag, step, value])


def _plot_training_curves(scalars: dict[str, list[tuple[int, float]]], output_file: Path) -> None:
    loss_tags = [tag for tag in ("metrics/train_loss", "metrics/val_loss") if tag in scalars]
    metric_tags = [tag for tag in ("metrics/p_opt", "metrics/p_exp", "metrics/h_mean") if tag in scalars]

    ncols = 2 if metric_tags else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    ax = axes[0]
    for tag in loss_tags:
        values = scalars[tag]
        ax.plot([step for step, _ in values], [value for _, value in values], marker="o", label=tag)
    ax.set_title("DDPM Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if loss_tags and max(len(scalars[tag]) for tag in loss_tags) < 3:
        ax.text(
            0.5,
            0.05,
            "too few points to judge convergence",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    if metric_tags:
        ax = axes[1]
        for tag in metric_tags:
            values = scalars[tag]
            ax.plot([step for step, _ in values], [value for _, value in values], marker="o", label=tag)
        ax.set_title("Validation A* Metrics")
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _parse_eval_log(eval_log: Path) -> list[dict[str, str | float]]:
    pattern = re.compile(
        r"^(?P<method>\S+)\s+"
        r"p_opt=(?P<p_opt>[0-9.]+)\s+"
        r"p_exp=(?P<p_exp>[0-9.]+)\s+"
        r"h_mean=(?P<h_mean>[0-9.]+)"
    )
    rows = []
    if not eval_log or not eval_log.exists():
        return rows
    for line in eval_log.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            row = {"method": match.group("method")}
            row.update({key: float(match.group(key)) for key in ("p_opt", "p_exp", "h_mean")})
            rows.append(row)
    return rows


def _write_eval_csv(rows: list[dict[str, str | float]], output_file: Path) -> None:
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "p_opt", "p_exp", "h_mean"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_eval(rows: list[dict[str, str | float]], output_file: Path) -> None:
    methods = [str(row["method"]) for row in rows]
    metrics = ["p_opt", "p_exp", "h_mean"]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for idx, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rows]
        ax.bar(x + (idx - 1) * width, values, width, label=metric)

    ax.set_title("Test Metrics")
    ax.set_xticks(x, methods)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--eval-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scalars = _load_scalars(args.event_dir)
    _write_scalars_csv(scalars, args.output_dir / "training_scalars.csv")
    _plot_training_curves(scalars, args.output_dir / "training_curves.png")
    print(f"wrote {args.output_dir / 'training_curves.png'}")

    rows = _parse_eval_log(args.eval_log) if args.eval_log else []
    if rows:
        _write_eval_csv(rows, args.output_dir / "eval_metrics.csv")
        _plot_eval(rows, args.output_dir / "eval_metrics.png")
        print(f"wrote {args.output_dir / 'eval_metrics.png'}")
    else:
        print("no eval metrics found; skipped eval plot")


if __name__ == "__main__":
    main()
