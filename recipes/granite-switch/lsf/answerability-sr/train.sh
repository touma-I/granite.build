#!/usr/bin/env bash
# Train a Shadow Residual answerability adapter.
#
# Run by custom_code_lsf as `start_command`, inside the activated conda env.
# Every value below arrives as an environment variable from build.yaml's
# byoc_config.env, which is where the parameters are documented.
#
# Two paths come from granite.build bindings rather than the recipe: TRAIN_DIR
# (the dataset repo pulled from the Hub, plus the subdirectory) and BASE (the
# base-model directory). Both are local directories by the time this runs.
#
# OUTPUT_PATH is exported by the step and already IS the dir_to_save directory
# (run_workload.sh: OUTPUT_PATH="${LLMB_LSF_OUTPUT_DIR}/${dir_to_save}"), so
# this writes straight into it. Writing to "$OUTPUT_PATH/adapter" instead —
# which the step's README suggests — would nest the artifact and upload nothing.
# The step echoes the LLMB_ARTIFACT_ID marker on success, so this must not.
set -euo pipefail

# Fail on an unset or empty variable here, naming it, rather than letting an
# empty path reach the trainer and surface as a confusing loader error.
: "${TRAIN_DIR:?}" "${BASE:?}" "${OUTPUT_PATH:?}" "${BASE_MODEL_ID:?}"
: "${MAX_LENGTH:?}" "${ASSISTANT_MARKER:?}" "${LAST_TOKEN:?}"
: "${LORA_RANK:?}" "${CROSS_STREAM_RANK:?}" "${LORA_ALPHA:?}" "${LORA_DROPOUT:?}"
: "${LEARNING_RATE:?}" "${WEIGHT_DECAY:?}" "${LR_SCHEDULER:?}" "${WARMUP_RATIO:?}"
: "${PER_DEVICE_TRAIN:?}" "${GRADIENT_ACCUMULATION:?}" "${PER_DEVICE_EVAL:?}"
: "${GRADIENT_CHECKPOINTING:?}" "${SAVE_TOTAL_LIMIT:?}" "${NUM_EPOCHS:?}"
: "${SEED:?}" "${LOGGING_STEPS:?}" "${DATALOADER_NUM_WORKERS:?}" "${REPORT_TO:?}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== inputs ==="
id -un
nvidia-smi -L || true
ls -la "$TRAIN_DIR" || true

# Cheapest possible probe that the bound directories actually arrived. A pull
# that landed on another node's filesystem fails here in seconds rather than
# minutes into from_pretrained.
test -f "$BASE/config.json" || {
    echo "ERROR: no config.json under $BASE — the base-model pull did not land here" >&2
    exit 1
}
for f in "$TRAIN_DIR/train.jsonl" "$TRAIN_DIR/test.jsonl"; do
    test -f "$f" || { echo "ERROR: missing $f — check TRAIN_DATA_SUBDIR" >&2; exit 1; }
done

# The config lands inside the artifact on purpose: the adapter directory then
# carries the exact configuration that produced it. PEFT ignores the extra file.
CFG="$OUTPUT_PATH/sr_answerability.yaml"
cat > "$CFG" <<YAML
model:
  base: ${BASE}
data:
  train_path: ${TRAIN_DIR}/train.jsonl
  val_path: ${TRAIN_DIR}/test.jsonl
  max_length: ${MAX_LENGTH}
  assistant_marker: "${ASSISTANT_MARKER}"
adapter:
  target_modules:
    q_proj: ${LORA_RANK}
    o_proj: ${LORA_RANK}
    gate_proj: ${LORA_RANK}
    up_proj: ${LORA_RANK}
    down_proj: ${LORA_RANK}
    cross_stream: ${CROSS_STREAM_RANK}
  alpha: ${LORA_ALPHA}
  dropout: ${LORA_DROPOUT}
  last_token: "${LAST_TOKEN}"
optimizer:
  name: adamw_torch
  learning_rate: ${LEARNING_RATE}
  weight_decay: ${WEIGHT_DECAY}
scheduler:
  type: ${LR_SCHEDULER}
  warmup_ratio: ${WARMUP_RATIO}
batch:
  per_device_train: ${PER_DEVICE_TRAIN}
  gradient_accumulation: ${GRADIENT_ACCUMULATION}
  per_device_eval: ${PER_DEVICE_EVAL}
  gradient_checkpointing: ${GRADIENT_CHECKPOINTING}
validation:
  strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  early_stopping_patience: 3
  stop_on_nan: true
save:
  strategy: epoch
  total_limit: ${SAVE_TOTAL_LIMIT}
  output_dir: ${OUTPUT_PATH}
runtime:
  num_train_epochs: ${NUM_EPOCHS}
  seed: ${SEED}
  logging_steps: ${LOGGING_STEPS}
  dataloader_num_workers: ${DATALOADER_NUM_WORKERS}
  report_to: ${REPORT_TO}
YAML

echo "=== generated SR training config ==="
cat "$CFG"

# MAX_STEPS > 0 turns the recipe into its own smoke test.
EXTRA=()
if [ "${MAX_STEPS:-0}" -gt 0 ]; then EXTRA+=(--max-steps "$MAX_STEPS"); fi

echo "=== training ==="
sr-train --config "$CFG" --skip-post-train-generate ${EXTRA[@]+"${EXTRA[@]}"}

test -f "$OUTPUT_PATH/adapter_config.json" || {
    echo "ERROR: training wrote no adapter_config.json to $OUTPUT_PATH" >&2
    exit 1
}

# PEFT records the path it loaded the base from, which here is a cluster-local
# cache directory and a dead reference anywhere else. Rewrite it to the Hub id
# so the adapter stays loadable off this cluster.
python - "$OUTPUT_PATH/adapter_config.json" "$BASE_MODEL_ID" <<'PY'
import json, sys
path, base_id = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = json.load(f)
cfg["base_model_name_or_path"] = base_id
with open(path, "w") as f:
    json.dump(cfg, f, indent=2)
PY

echo "=== adapter base recorded as ==="
grep base_model_name_or_path "$OUTPUT_PATH/adapter_config.json"
ls -la "$OUTPUT_PATH"
