#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p experiments/alignment/logs experiments/alignment/runs

run_one() {
    local method="$1"
    local seed="$2"
    local output_dir="experiments/alignment/runs/${method}_full_seed${seed}"
    local log_file="experiments/alignment/logs/${method}_full_seed${seed}.log"

    if [[ -f "$output_dir/metrics.jsonl" ]]; then
        echo "ERROR: run already exists: $output_dir"
        exit 1
    fi

    mkdir -p "$output_dir"

    echo "$log_file" \
      > experiments/alignment/current_multiseed_log.txt

    echo "============================================================"
    echo "START: method=$method seed=$seed"
    echo "TIME:  $(date)"
    echo "LOG:   $log_file"
    echo "============================================================"

    common_args=(
        uv run --no-sync python -u scripts/alignment/train_grpo.py
        --prompt-file src/llm_training_scaling/alignment/prompts/r1_zero.prompt
        --output-dir "$output_dir"
        --seed "$seed"
        --num-steps 200
        --n-train-examples 6400
        --n-val-examples 1024
        --rollout-batch-size 256
        --group-size 8
        --learning-rate 1e-5
        --max-grad-norm 1.0
        --sampling-temperature 1.0
        --sampling-max-tokens 512
        --eval-every 10
        --generation-batch-size 32
        --log-rollouts-every 20
        --baseline mean
    )

    case "$method" in
        grpo_standard)
            extra_args=(
                --gradient-accumulation-steps 32
                --advantage-normalizer std
                --loss-normalization sequence
                --importance-reweighting-method none
            )
            ;;

        dr_grpo)
            extra_args=(
                --gradient-accumulation-steps 32
                --advantage-normalizer none
                --loss-normalization constant
                --normalization-constant 131072
                --importance-reweighting-method none
            )
            ;;

        offpolicy_clip)
            extra_args=(
                --train-batch-size 8
                --gradient-accumulation-steps 1
                --advantage-normalizer std
                --loss-normalization sequence
                --importance-reweighting-method grpo
                --cliprange 0.2
            )
            ;;

        *)
            echo "Unknown method: $method"
            exit 1
            ;;
    esac

    "${common_args[@]}" "${extra_args[@]}" \
        > "$log_file" 2>&1

    echo "============================================================"
    echo "DONE:  method=$method seed=$seed"
    echo "TIME:  $(date)"
    echo "============================================================"
}

# 每个 seed 先完成三种方法，便于尽早得到完整的配对结果。
for seed in 1 2 3; do
    run_one grpo_standard "$seed"
    run_one dr_grpo "$seed"
    run_one offpolicy_clip "$seed"
done

echo "ALL MULTI-SEED RUNS COMPLETED"
echo "TIME: $(date)"
