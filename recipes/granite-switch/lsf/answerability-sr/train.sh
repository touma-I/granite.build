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
: "${MAX_LENGTH:?}" "${ASSISTANT_MARKER:?}" "${LAST_TOKEN:?}" "${LAST_CONTEXT_TOKEN:?}"
: "${LORA_RANK:?}" "${CROSS_STREAM_RANK:?}" "${LORA_ALPHA:?}" "${LORA_DROPOUT:?}"
: "${LEARNING_RATE:?}" "${WEIGHT_DECAY:?}" "${LR_SCHEDULER:?}" "${WARMUP_RATIO:?}"
: "${PER_DEVICE_TRAIN:?}" "${GRADIENT_ACCUMULATION:?}" "${PER_DEVICE_EVAL:?}"
: "${GRADIENT_CHECKPOINTING:?}" "${SAVE_TOTAL_LIMIT:?}" "${NUM_EPOCHS:?}"
: "${SEED:?}" "${LOGGING_STEPS:?}" "${DATALOADER_NUM_WORKERS:?}" "${REPORT_TO:?}"

# The marker is the one value here carrying a backslash escape, and it passes
# through two YAML parses and a line-based env loader on the way (see the
# comment beside it in build.yaml). Each of those can eat the escape: a real
# newline breaks the step's loader outright, while a dropped one trains against
# a silently mis-anchored loss mask. Check the literal two characters survived.
case "$ASSISTANT_MARKER" in
  *'\n<think></think>') : ;;
  *) echo "ERROR: ASSISTANT_MARKER arrived as $(printf '%q' "$ASSISTANT_MARKER")," >&2
     echo "       expected it to end in the two literal characters \\n followed by" >&2
     echo "       <think></think>. The escape was consumed in transit — check the" >&2
     echo "       quoting of ASSISTANT_MARKER in build.yaml's byoc_config.env." >&2
     exit 1 ;;
esac

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

# Deliberately NOT written inside $OUTPUT_PATH yet. The step emits its
# artifact event whenever $OUTPUT_PATH has any content "even on failure"
# (run_workload.sh), so a stray file there turns a failed run into a published
# artifact and lets the eval target start against an adapter that was never
# trained. It is copied in at the end, once there is something to describe.
CFG="$(mktemp -d)/sr_answerability.yaml"
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
  last_context_token: "${LAST_CONTEXT_TOKEN}"
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

# `python -m`, NOT the `sr-train` console script. The step builds the conda env
# at a temporary prefix and moves it to its final path, while pip bakes an
# absolute interpreter path into console-script shebangs at install time — so
# `sr-train` points at an interpreter that no longer exists and dies with exit
# 126, "bad interpreter". Module invocation resolves python at runtime.
# sr-train is exactly shadow_residual.training.train:main, and that module has
# a __main__ guard, so this is the same entry point.
echo "=== training ==="
python -m shadow_residual.training.train --config "$CFG" --skip-post-train-generate ${EXTRA[@]+"${EXTRA[@]}"}

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

# Now that there is a real adapter, keep the config beside it: the artifact
# then carries the exact configuration that produced it. PEFT ignores the file.
cp "$CFG" "$OUTPUT_PATH/sr_answerability.yaml"

echo "=== adapter base recorded as ==="
grep base_model_name_or_path "$OUTPUT_PATH/adapter_config.json"
ls -la "$OUTPUT_PATH"
