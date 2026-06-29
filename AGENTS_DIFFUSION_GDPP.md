# GDPP Diffusion A* Agent Guide

This repository now contains a GDPP-style diffusion first stage for neural-astar.
Use this document when an agent needs to train the method from scratch and run
the comparison evaluation.

## What Changed

The new method keeps neural-astar's second stage unchanged:

```text
map + start + goal
  -> conditional DDPM U-Net
  -> bright path heatmap
  -> cost map = 1 - heatmap
  -> existing DifferentiableAstar
  -> p_opt / p_exp / h_mean evaluation
```

Important files:

- `src/neural_astar/planner/diffusion.py`: DDPM scheduler ported from GDPP.
- `src/neural_astar/planner/diffusion_unet.py`: GDPP conditional U-Net, fully convolutional and compatible with 32x32 maps.
- `src/neural_astar/utils/heatmap.py`: Gaussian mask-to-heatmap helper.
- `src/neural_astar/planner/diffusion_astar.py`: `DiffusionAstar`, which samples a heatmap and flips it into a cost map.
- `src/neural_astar/planner/gdpp_no_mse_astar.py`: `GDPPNoMSEAstar`, which runs GDPP heatmap-cost A* and skips render-MSE.
- `src/neural_astar/utils/training.py`: `DiffusionPlannerModule`, which trains DDPM noise prediction.
- `scripts/train_diffusion.py`: Hydra training entry point.
- `scripts/eval_compare.py`: VanillaAstar / NeuralAstar / DiffusionAstar comparison.
- `scripts/run_diffusion_from_scratch_eval.sh`: one-command train-from-scratch and evaluate runner.
- `scripts/plot_diffusion_results.py`: training curve and eval metric plots.
- `scripts/visualize_diffusion_outputs.py`: per-scene map, cost, heatmap, path, and history panels.

## Environment

Run from the repository root:

```bash
cd /Users/yulinli/neural-astar
```

The Python environment must provide:

```text
torch
pytorch_lightning
hydra-core
omegaconf
matplotlib
tensorboard
```

If dependencies are missing, create or activate an environment before running
the script. The repository's `pyproject.toml` lists the original neural-astar
dependency set, but newer local PyTorch versions also work for the smoke tests.

## Dataset

The default dataset is:

```text
planning-datasets/data/mpd/mazes_032_moore_c8.npz
```

The runner initializes the submodule if the file is missing:

```bash
git submodule update --init planning-datasets
```

## Full Run

Default full run:

```bash
bash scripts/run_diffusion_from_scratch_eval.sh
```

This trains a fresh diffusion model from random initialization. The default is
longer training with early stopping:

```text
max epochs: 500
early stop monitor: metrics/val_diffusion_loss
early stop mode: min
early stop patience: 80 validation epochs
checkpoint monitor: metrics/h_mean
checkpoint mode: max
```

The runner evaluates the printed `best_model_path`, not merely the latest
checkpoint. It then compares:

- `VanillaAstar`
- `NeuralAstar`, if `NEURAL_CKPT` exists
- `DiffusionAstar`, using the newly trained checkpoint and neural-astar's DifferentiableAstar
- `GDPPNoMSE`, using the same checkpoint with GDPP-style plain A* and no render-MSE

Default output locations:

```text
model/gdpp_diffusion_from_scratch/mazes_032_moore_c8_diffusion/
outputs/gdpp_diffusion_from_scratch/<timestamp>/
```

The run directory contains:

```text
train.log
eval.log
diffusion_ckpt.txt
summary.env
plots/training_curves.png
plots/training_scalars.csv
plots/eval_metrics.png
plots/eval_metrics.csv
visualizations/contact_sheet.png
visualizations/per_sample_metrics.csv
visualizations/samples/sample_*.png
```

## Useful Overrides

Short smoke run:

```bash
EPOCHS=1 \
BATCH_SIZE=2 \
DIFFUSION_STEPS=2 \
BASE_CHANNELS=8 \
TIME_EMB_DIM=32 \
CHANNEL_MULTS='[1,2,4]' \
TRAIN_EXTRA_OVERRIDES='params.limit_train_batches=1 params.limit_val_batches=1' \
EVAL_BATCH_SIZE=2 \
EVAL_MAX_BATCHES=1 \
bash scripts/run_diffusion_from_scratch_eval.sh
```

Longer training with default 32x32-compatible U-Net:

```bash
EPOCHS=500 BATCH_SIZE=100 DIFFUSION_STEPS=100 \
bash scripts/run_diffusion_from_scratch_eval.sh
```

Four-GPU DDP example:

```bash
DEVICES=4 STRATEGY=ddp EPOCHS=500 EARLY_STOP_PATIENCE=80 \
bash scripts/run_diffusion_from_scratch_eval.sh
```

Use a specific Python:

```bash
PYTHON=/path/to/venv/bin/python bash scripts/run_diffusion_from_scratch_eval.sh
```

Use a specific NeuralAstar baseline checkpoint:

```bash
NEURAL_CKPT='model/mazes_032_moore_c8/lightning_logs/version_0/checkpoints/epoch=33-step=272.ckpt' \
bash scripts/run_diffusion_from_scratch_eval.sh
```

Skip NeuralAstar comparison:

```bash
NEURAL_CKPT=null bash scripts/run_diffusion_from_scratch_eval.sh
```

## Direct Commands

Train only:

```bash
PYTHONPATH=src python scripts/train_diffusion.py \
  dataset=planning-datasets/data/mpd/mazes_032_moore_c8 \
  logdir=model/gdpp_diffusion_from_scratch \
  params.num_epochs=50 \
  params.batch_size=100
```

Evaluate only:

```bash
PYTHONPATH=src python scripts/eval_compare.py \
  dataset=planning-datasets/data/mpd/mazes_032_moore_c8 \
  'neural_ckpt=model/mazes_032_moore_c8/lightning_logs/version_0/checkpoints/epoch\=33-step\=272.ckpt' \
  'diffusion_ckpt=/path/to/diffusion.ckpt'
```

Hydra needs `=` characters inside checkpoint filenames escaped as `\=`.
The runner handles this automatically.

Plot an existing run:

```bash
python scripts/plot_diffusion_results.py \
  --event-dir model/gdpp_diffusion_from_scratch/mazes_032_moore_c8_diffusion/lightning_logs/version_0 \
  --eval-log outputs/gdpp_diffusion_from_scratch/<timestamp>/eval.log \
  --output-dir outputs/gdpp_diffusion_from_scratch/<timestamp>/plots
```

Visualize 200 test scenes from an existing checkpoint:

```bash
python scripts/visualize_diffusion_outputs.py \
  --dataset planning-datasets/data/mpd/mazes_032_moore_c8.npz \
  --output-dir outputs/gdpp_diffusion_from_scratch/<timestamp>/visualizations \
  --num-samples 200 \
  --neural-ckpt 'model/mazes_032_moore_c8/lightning_logs/version_0/checkpoints/epoch=33-step=272.ckpt' \
  --diffusion-ckpt '/path/to/diffusion.ckpt'
```

The visualization panels include map/ground-truth path, VanillaAstar path,
NeuralAstar cost/path, diffusion heatmap, `1 - heatmap` cost/path, and GDPP
`-log(heatmap + eps)` cost/path.

## Notes for Future Agents

- `DiffusionPlannerModule.training_step` does not run A*. It trains the DDPM
  denoising objective on a Gaussianized `opt_traj + goal_map` target.
- Validation logs both `metrics/val_diffusion_loss` for convergence/early-stop
  and A* metrics (`metrics/p_opt`, `metrics/p_exp`, `metrics/h_mean`) for model
  selection.
- Validation and evaluation do run the existing `DifferentiableAstar`.
- `GDPPNoMSE` is the exception: it evaluates the same diffusion heatmap with
  GDPP's `-log(heatmap + eps)` cost map and plain 8-neighbor A*, then stops
  before render-MSE optimization.
- `map_design` uses neural-astar polarity: `1` means passable and `0` means blocked.
- The diffusion target is bright on the path; inference flips it with
  `cost = 1 - heatmap`.
- Do not compare against GDPP's render-MSE second stage here. This integration is
  intentionally a first-stage replacement only.
