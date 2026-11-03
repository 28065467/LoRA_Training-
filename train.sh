#!/usr/bin/env bash
# CLI wrapper around scripts/run_training_pipeline.py — sets its CONFIG via env vars instead
# of making you edit the Python file. Any field you don't pass keeps that script's own default
# (see scripts/run_training_pipeline.py's CONFIG block).
#
# Usage:
#   ./train.sh --train data/train.json --val data/val.json --test data/test.json \
#              --output-dir outputs/my-run [options...]
#
# Required:
#   --train PATH          TRAIN_DATA_PATH (chatml json)
#   --val PATH             VAL_DATA_PATH
#   --test PATH            TEST_DATA_PATH
#   --output-dir PATH      OUTPUT_DIR — use a fresh directory per run, never reuse one
#
# Optional (defaults shown are run_training_pipeline.py's own defaults):
#   --model-id ID          MODEL_ID                  (default: Qwen/Qwen3-Coder-30B-A3B-Instruct)
#   --gpu N                CUDA_VISIBLE_DEVICES       (default: 0)
#   --conda-env NAME       CONDA_ENV                  (default: nemo)
#   --epochs N             NUM_EPOCHS                 (default: 2)
#   --dim N                LORA_DIM                   (default: 32)
#   --alpha N               LORA_ALPHA                 (default: 64)
#   --dropout F            LORA_DROPOUT               (default: 0.05)
#   --lr F                  LEARNING_RATE              (default: 1e-4)
#   --weight-decay F       WEIGHT_DECAY               (default: 0.05)
#   --seq-length N          SEQ_LENGTH / PROMPT_TOKEN_LIMIT (default: 6144)
#   --wsd-decay-steps N    WSD_DECAY_STEPS            (default: 80 — see note below)
#   --local-batch N        LOCAL_BATCH_SIZE           (default: 4)
#   --global-batch N       GLOBAL_BATCH_SIZE          (default: 16)
#   --eval-checkpoint NAME EVAL_CHECKPOINT            (default: LATEST; or LOWEST_VAL, epoch_N_step_M)
#   --no-fresh-start        FRESH_START=false (resume instead of archiving existing checkpoints)
#   --tmux NAME             launch inside a new tmux session of this name instead of foreground
#
# Small-dataset note: if --train has only a few hundred examples, the default epochs/decay-steps
# combo can give a degenerate LR schedule. Raise --epochs and shrink --wsd-decay-steps so decay
# steps land around 15-20% of total steps (total steps ≈ epochs * len(train) / global_batch).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="nemo"
TMUX_SESSION=""
declare -A ENV_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) ENV_OVERRIDES[MODEL_ID]="$2"; shift 2 ;;
    --train) ENV_OVERRIDES[TRAIN_DATA_PATH]="$2"; shift 2 ;;
    --val) ENV_OVERRIDES[VAL_DATA_PATH]="$2"; shift 2 ;;
    --test) ENV_OVERRIDES[TEST_DATA_PATH]="$2"; shift 2 ;;
    --output-dir) ENV_OVERRIDES[OUTPUT_DIR]="$2"; shift 2 ;;
    --gpu) ENV_OVERRIDES[CUDA_VISIBLE_DEVICES]="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --epochs) ENV_OVERRIDES[NUM_EPOCHS]="$2"; shift 2 ;;
    --dim) ENV_OVERRIDES[LORA_DIM]="$2"; shift 2 ;;
    --alpha) ENV_OVERRIDES[LORA_ALPHA]="$2"; shift 2 ;;
    --dropout) ENV_OVERRIDES[LORA_DROPOUT]="$2"; shift 2 ;;
    --lr) ENV_OVERRIDES[LEARNING_RATE]="$2"; shift 2 ;;
    --weight-decay) ENV_OVERRIDES[WEIGHT_DECAY]="$2"; shift 2 ;;
    --seq-length)
      ENV_OVERRIDES[SEQ_LENGTH]="$2"; ENV_OVERRIDES[PROMPT_TOKEN_LIMIT]="$2"; shift 2 ;;
    --wsd-decay-steps) ENV_OVERRIDES[WSD_DECAY_STEPS]="$2"; shift 2 ;;
    --local-batch) ENV_OVERRIDES[LOCAL_BATCH_SIZE]="$2"; shift 2 ;;
    --global-batch) ENV_OVERRIDES[GLOBAL_BATCH_SIZE]="$2"; shift 2 ;;
    --eval-checkpoint) ENV_OVERRIDES[EVAL_CHECKPOINT]="$2"; shift 2 ;;
    --no-fresh-start) ENV_OVERRIDES[FRESH_START]="false"; shift ;;
    --tmux) TMUX_SESSION="$2"; shift 2 ;;
    -h|--help) grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for required in TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH OUTPUT_DIR; do
  if [[ -z "${ENV_OVERRIDES[$required]:-}" ]]; then
    echo "Missing required option for $required (see --help)." >&2
    exit 1
  fi
done

mkdir -p "${ENV_OVERRIDES[OUTPUT_DIR]}"
LOG_PATH="${ENV_OVERRIDES[OUTPUT_DIR]}/training_pipeline.log"

# Build the env-var assignments + command as one string so it can run either directly or
# inside a tmux session the same way.
ENV_ASSIGNMENTS=""
for key in "${!ENV_OVERRIDES[@]}"; do
  ENV_ASSIGNMENTS+="export ${key}=$(printf '%q' "${ENV_OVERRIDES[$key]}"); "
done

RUN_CMD="source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate ${CONDA_ENV} && \
${ENV_ASSIGNMENTS} \
cd \"${SCRIPT_DIR}\" && \
python -u scripts/run_training_pipeline.py 2>&1 | tee \"${LOG_PATH}\""

if [[ -n "$TMUX_SESSION" ]]; then
  TMUX_BIN="$(command -v tmux || echo "$(conda info --base)/bin/tmux")"
  "$TMUX_BIN" new-session -d -s "$TMUX_SESSION" "$RUN_CMD"
  echo "Launched in tmux session '$TMUX_SESSION'. Attach with: tmux attach -t $TMUX_SESSION"
  echo "Log: $LOG_PATH"
else
  eval "$RUN_CMD"
fi
