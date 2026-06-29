#!/usr/bin/env bash
set -euo pipefail

# Train GDPP-style diffusion cost-map generation from scratch on neural-astar's
# mazes_032 dataset, then evaluate against VanillaAstar and the NeuralAstar
# checkpoint when available.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python3}"
DATASET="${DATASET:-planning-datasets/data/mpd/mazes_032_moore_c8}"
DATASET_FILE="${DATASET}.npz"
DATASET_BASENAME="$(basename "$DATASET")"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/gdpp_diffusion_from_scratch/${RUN_ID}}"
LOGDIR="${LOGDIR:-model/gdpp_diffusion_from_scratch}"
TRAIN_ROOT="${LOGDIR}/${DATASET_BASENAME}_diffusion"
RUN_MARKER="${RUN_DIR}/run.marker"

NEURAL_CKPT_DEFAULT="model/mazes_032_moore_c8/lightning_logs/version_0/checkpoints/epoch=33-step=272.ckpt"
NEURAL_CKPT="${NEURAL_CKPT:-$NEURAL_CKPT_DEFAULT}"

EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-100}"
LR="${LR:-0.0001}"
TMAX="${TMAX:-0.25}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
BETA_SCHEDULE="${BETA_SCHEDULE:-cosine}"
UNET_MODEL_TYPE="${UNET_MODEL_TYPE:-baseline}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
TIME_EMB_DIM="${TIME_EMB_DIM:-256}"
CHANNEL_MULTS="${CHANNEL_MULTS:-[1,2,4,8]}"
TRAJECTORY_SIGMA="${TRAJECTORY_SIGMA:-1.0}"
START_GOAL_SIGMA="${START_GOAL_SIGMA:-1.0}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-100}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-null}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
VISUALIZE_SAMPLES="${VISUALIZE_SAMPLES:-200}"
VISUALIZE_CONTACT_SHEET_MAX="${VISUALIZE_CONTACT_SHEET_MAX:-32}"
VISUALIZE_DEVICE="${VISUALIZE_DEVICE:-$EVAL_DEVICE}"

EARLY_STOP="${EARLY_STOP:-true}"
EARLY_STOP_MONITOR="${EARLY_STOP_MONITOR:-metrics/val_diffusion_loss}"
EARLY_STOP_MODE="${EARLY_STOP_MODE:-min}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-80}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-metrics/h_mean}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
DEVICES="${DEVICES:-null}"
STRATEGY="${STRATEGY:-null}"
PRECISION="${PRECISION:-null}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"

# Optional Hydra override fragments appended verbatim.
TRAIN_EXTRA_OVERRIDES="${TRAIN_EXTRA_OVERRIDES:-}"
EVAL_EXTRA_OVERRIDES="${EVAL_EXTRA_OVERRIDES:-}"

mkdir -p "$RUN_DIR"
touch "$RUN_MARKER"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

hydra_escape_value() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//=/\\=}"
  printf '%s' "$value"
}

require_python_imports() {
  "$PYTHON_BIN" - <<'PY'
missing = []
for module in ("torch", "pytorch_lightning", "hydra", "omegaconf"):
    try:
        __import__(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")
if missing:
    raise SystemExit("Missing Python dependencies:\n  " + "\n  ".join(missing))
PY
}

echo "== GDPP diffusion from-scratch run =="
echo "root:        $ROOT"
echo "python:      $PYTHON_BIN"
echo "dataset:     $DATASET_FILE"
echo "run_dir:     $RUN_DIR"
echo "logdir:      $LOGDIR"
echo "train_root:  $TRAIN_ROOT"
echo "epochs:      $EPOCHS"
echo "early_stop:  $EARLY_STOP_MONITOR ($EARLY_STOP_MODE, patience=$EARLY_STOP_PATIENCE)"

if [[ ! -f "$DATASET_FILE" ]]; then
  echo "Dataset not found; initializing planning-datasets submodule..."
  git submodule update --init planning-datasets
fi

if [[ ! -f "$DATASET_FILE" ]]; then
  echo "ERROR: dataset still missing: $DATASET_FILE" >&2
  exit 1
fi

require_python_imports

if [[ ! -f "$NEURAL_CKPT" ]]; then
  echo "WARNING: NeuralAstar checkpoint not found; eval will skip NeuralAstar: $NEURAL_CKPT"
  NEURAL_CKPT="null"
fi

train_overrides=(
  "dataset=$DATASET"
  "logdir=$LOGDIR"
  "Tmax=$TMAX"
  "params.num_epochs=$EPOCHS"
  "params.batch_size=$BATCH_SIZE"
  "params.lr=$LR"
  "diffusion.num_steps=$DIFFUSION_STEPS"
  "diffusion.beta_schedule=$BETA_SCHEDULE"
  "diffusion.trajectory_sigma=$TRAJECTORY_SIGMA"
  "diffusion.start_goal_sigma=$START_GOAL_SIGMA"
  "unet.model_type=$UNET_MODEL_TYPE"
  "unet.base_channels=$BASE_CHANNELS"
  "unet.time_emb_dim=$TIME_EMB_DIM"
  "unet.channel_mults=$CHANNEL_MULTS"
  "checkpoint.monitor=$CHECKPOINT_MONITOR"
  "checkpoint.mode=$CHECKPOINT_MODE"
  "early_stop.enabled=$EARLY_STOP"
  "early_stop.monitor=$EARLY_STOP_MONITOR"
  "early_stop.mode=$EARLY_STOP_MODE"
  "early_stop.patience=$EARLY_STOP_PATIENCE"
  "early_stop.min_delta=$EARLY_STOP_MIN_DELTA"
  "trainer.devices=$DEVICES"
  "trainer.strategy=$STRATEGY"
  "trainer.precision=$PRECISION"
  "trainer.accumulate_grad_batches=$ACCUMULATE_GRAD_BATCHES"
  "trainer.check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH"
)

if [[ -n "$TRAIN_EXTRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  train_overrides+=( $TRAIN_EXTRA_OVERRIDES )
fi

echo
echo "== Training diffusion planner from scratch =="
"$PYTHON_BIN" scripts/train_diffusion.py "${train_overrides[@]}" 2>&1 | tee "$RUN_DIR/train.log"

DIFFUSION_CKPT="$(grep -E '^best_model_path=' "$RUN_DIR/train.log" | tail -n 1 | cut -d= -f2-)"
if [[ -z "$DIFFUSION_CKPT" || ! -f "$DIFFUSION_CKPT" ]]; then
  DIFFUSION_CKPT="$(
  "$PYTHON_BIN" - "$TRAIN_ROOT" "$RUN_MARKER" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
marker = pathlib.Path(sys.argv[2])
marker_time = marker.stat().st_mtime
checkpoints = [
    path for path in root.rglob("*.ckpt")
    if path.stat().st_mtime >= marker_time
]
if checkpoints:
    print(max(checkpoints, key=lambda path: path.stat().st_mtime))
PY
  )"
fi

if [[ -z "$DIFFUSION_CKPT" || ! -f "$DIFFUSION_CKPT" ]]; then
  echo "ERROR: no new diffusion checkpoint found under $TRAIN_ROOT" >&2
  exit 1
fi

echo "$DIFFUSION_CKPT" > "$RUN_DIR/diffusion_ckpt.txt"
echo "diffusion_ckpt: $DIFFUSION_CKPT"

neural_ckpt_override="neural_ckpt=$NEURAL_CKPT"
if [[ "$NEURAL_CKPT" != "null" ]]; then
  neural_ckpt_override="neural_ckpt=$(hydra_escape_value "$NEURAL_CKPT")"
fi

eval_overrides=(
  "dataset=$DATASET"
  "batch_size=$EVAL_BATCH_SIZE"
  "max_batches=$EVAL_MAX_BATCHES"
  "device=$EVAL_DEVICE"
  "$neural_ckpt_override"
  "diffusion_ckpt=$(hydra_escape_value "$DIFFUSION_CKPT")"
  "diffusion_model.diffusion.num_steps=$DIFFUSION_STEPS"
  "diffusion_model.diffusion.beta_schedule=$BETA_SCHEDULE"
  "diffusion_model.unet.model_type=$UNET_MODEL_TYPE"
  "diffusion_model.unet.base_channels=$BASE_CHANNELS"
  "diffusion_model.unet.time_emb_dim=$TIME_EMB_DIM"
  "diffusion_model.unet.channel_mults=$CHANNEL_MULTS"
  "diffusion_model.start_goal_sigma=$START_GOAL_SIGMA"
)

if [[ -n "$EVAL_EXTRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  eval_overrides+=( $EVAL_EXTRA_OVERRIDES )
fi

echo
echo "== Evaluating VanillaAstar / NeuralAstar / DiffusionAstar / GDPPNoMSE =="
"$PYTHON_BIN" scripts/eval_compare.py "${eval_overrides[@]}" 2>&1 | tee "$RUN_DIR/eval.log"

LIGHTNING_RUN_DIR="$(dirname "$(dirname "$DIFFUSION_CKPT")")"
PLOT_DIR="$RUN_DIR/plots"
echo
echo "== Plotting training/evaluation curves =="
if "$PYTHON_BIN" scripts/plot_diffusion_results.py \
  --event-dir "$LIGHTNING_RUN_DIR" \
  --eval-log "$RUN_DIR/eval.log" \
  --output-dir "$PLOT_DIR"; then
  echo "plots: $PLOT_DIR"
else
  echo "WARNING: plotting failed; install matplotlib and tensorboard to generate plots" >&2
fi

VISUALIZE_DIR="$RUN_DIR/visualizations"
if [[ "$VISUALIZE_SAMPLES" != "0" ]]; then
  echo
  echo "== Visualizing planner outputs (${VISUALIZE_SAMPLES} samples) =="
  # Convert Hydra-style "[1,2,4,8]" into argparse tokens: 1 2 4 8.
  # shellcheck disable=SC2207
  CHANNEL_MULTS_TOKENS=($(printf '%s' "$CHANNEL_MULTS" | tr -d '[]' | tr ',' ' '))
  visualize_args=(
    --dataset "$DATASET_FILE"
    --split test
    --output-dir "$VISUALIZE_DIR"
    --num-samples "$VISUALIZE_SAMPLES"
    --device "$VISUALIZE_DEVICE"
    --diffusion-ckpt "$DIFFUSION_CKPT"
    --diffusion-steps "$DIFFUSION_STEPS"
    --beta-schedule "$BETA_SCHEDULE"
    --diffusion-model-type "$UNET_MODEL_TYPE"
    --base-channels "$BASE_CHANNELS"
    --time-emb-dim "$TIME_EMB_DIM"
    --channel-mults "${CHANNEL_MULTS_TOKENS[@]}"
    --start-goal-sigma "$START_GOAL_SIGMA"
    --gdpp-lam 1.0
    --gdpp-heuristic-weight 1.0
    --contact-sheet-max "$VISUALIZE_CONTACT_SHEET_MAX"
  )
  if [[ "$NEURAL_CKPT" != "null" ]]; then
    visualize_args+=(--neural-ckpt "$NEURAL_CKPT")
  fi
  if "$PYTHON_BIN" scripts/visualize_diffusion_outputs.py "${visualize_args[@]}" 2>&1 | tee "$RUN_DIR/visualize.log"; then
    echo "visualizations: $VISUALIZE_DIR"
  else
    echo "WARNING: visualization failed; install matplotlib and ensure checkpoints/configs match" >&2
  fi
fi

{
  echo "run_dir=$RUN_DIR"
  echo "dataset=$DATASET_FILE"
  echo "diffusion_ckpt=$DIFFUSION_CKPT"
  echo "neural_ckpt=$NEURAL_CKPT"
  echo "train_log=$RUN_DIR/train.log"
  echo "eval_log=$RUN_DIR/eval.log"
  echo "plots=$PLOT_DIR"
  echo "visualizations=$VISUALIZE_DIR"
} > "$RUN_DIR/summary.env"

echo
echo "== Done =="
cat "$RUN_DIR/summary.env"
