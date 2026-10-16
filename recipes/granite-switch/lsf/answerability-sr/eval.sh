#!/usr/bin/env bash
# Score a trained Shadow Residual answerability adapter.
#
# Run by custom_code_lsf as `start_command`, inside the activated conda env.
# ADAPTER comes from the train target's output artifact via a binding, so this
# target cannot start until that artifact exists; BASE is the same base-model
# artifact train bound, which is what makes "scored on the base it was trained
# on" a recorded fact rather than a convention.
#
# As in train.sh, OUTPUT_PATH already IS the dir_to_save directory, and the
# step emits the LLMB_ARTIFACT_ID marker itself.
set -euo pipefail

: "${ADAPTER:?}" "${EVAL_DIR:?}" "${BASE:?}" "${OUTPUT_PATH:?}"
: "${EVAL_DATASET_SIZE:?}" "${EVAL_BATCH_SIZE:?}" "${EVAL_MAX_NEW_TOKENS:?}"

export PYTHONUNBUFFERED=1

echo "=== inputs ==="
id -un
nvidia-smi -L || true
ls -la "$ADAPTER" || true
ls -la "$EVAL_DIR" || true

test -f "$BASE/config.json" || {
    echo "ERROR: no config.json under $BASE — the base-model pull did not land here" >&2
    exit 1
}
test -f "$ADAPTER/adapter_config.json" || {
    echo "ERROR: no adapter_config.json under $ADAPTER" >&2
    exit 1
}

echo "=== evaluating ==="
python -m shadow_residual.eval.answerability_eval \
    --mode peft \
    --base-model "$BASE" \
    --adapter-path "$ADAPTER" \
    --adapter-sub-path . \
    --dataset-path "$EVAL_DIR" \
    --dataset-size "$EVAL_DATASET_SIZE" \
    --output-path "$OUTPUT_PATH" \
    --device cuda \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS"

test -f "$OUTPUT_PATH/eval_summary.md" || {
    echo "ERROR: eval wrote no eval_summary.md to $OUTPUT_PATH" >&2
    exit 1
}

echo "=== eval summary ==="
cat "$OUTPUT_PATH/eval_summary.md"
