#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p experiments/alignment/runs/dr_grpo_full_seed0 experiments/alignment/logs

exec uv run --no-sync python -u scripts/alignment/train_grpo.py \
  --prompt-file src/llm_training_scaling/alignment/prompts/r1_zero.prompt \
  --output-dir experiments/alignment/runs/dr_grpo_full_seed0 \
  --seed 0 \
  --num-steps 200 \
  --n-train-examples 6400 \
  --n-val-examples 1024 \
  --rollout-batch-size 256 \
  --group-size 8 \
  --gradient-accumulation-steps 32 \
  --learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --sampling-temperature 1.0 \
  --sampling-max-tokens 512 \
  --eval-every 10 \
  --generation-batch-size 32 \
  --log-rollouts-every 20 \
  --baseline mean \
  --advantage-normalizer none \
  --loss-normalization constant \
  --normalization-constant 131072 \
  --save-final
