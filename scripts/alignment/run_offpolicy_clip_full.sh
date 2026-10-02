#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="experiments/alignment/runs/offpolicy_clip_full_seed0"
LOG_FILE="experiments/alignment/logs/offpolicy_clip_full_seed0.log"

mkdir -p experiments/alignment/logs

if [[ -f "$OUTPUT_DIR/metrics.jsonl" ]]; then
    echo "Refusing to append to existing run: $OUTPUT_DIR/metrics.jsonl"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

exec uv run --no-sync python -u scripts/alignment/train_grpo.py \
  --prompt-file src/llm_training_scaling/alignment/prompts/r1_zero.prompt \
  --output-dir "$OUTPUT_DIR" \
  --seed 0 \
  --num-steps 200 \
  --n-train-examples 6400 \
  --n-val-examples 1024 \
  --rollout-batch-size 256 \
  --train-batch-size 8 \
  --group-size 8 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --sampling-temperature 1.0 \
  --sampling-max-tokens 512 \
  --eval-every 10 \
  --generation-batch-size 32 \
  --log-rollouts-every 20 \
  --baseline mean \
  --advantage-normalizer std \
  --loss-normalization sequence \
  --importance-reweighting-method grpo \
  --cliprange 0.2 \
  --save-final
