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

EPOCHS="${EPOCHS:-50}"
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
)

if [[ -n "$TRAIN_EXTRA_OVERRIDES" ]]; then
  # shellcheck disable=SC2206
  train_overrides+=( $TRAIN_EXTRA_OVERRIDES )
fi

echo
echo "== Training diffusion planner from scratch =="
"$PYTHON_BIN" scripts/train_diffusion.py "${train_overrides[@]}" 2>&1 | tee "$RUN_DIR/train.log"

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
echo "== Evaluating VanillaAstar / NeuralAstar / DiffusionAstar =="
"$PYTHON_BIN" scripts/eval_compare.py "${eval_overrides[@]}" 2>&1 | tee "$RUN_DIR/eval.log"

{
  echo "run_dir=$RUN_DIR"
  echo "dataset=$DATASET_FILE"
  echo "diffusion_ckpt=$DIFFUSION_CKPT"
  echo "neural_ckpt=$NEURAL_CKPT"
  echo "train_log=$RUN_DIR/train.log"
  echo "eval_log=$RUN_DIR/eval.log"
} > "$RUN_DIR/summary.env"

echo
echo "== Done =="
cat "$RUN_DIR/summary.env"
