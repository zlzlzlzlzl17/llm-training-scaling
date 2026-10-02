# Experiment methodology

## Systems

Benchmarks use CUDA event timing after warmup, report mean and standard deviation, and record GPU memory where relevant. Distributed experiments use two GPUs and compare implementations at a fixed model/batch configuration.

## Scaling

Training reads each token stream deterministically from left to right. Every run uses the same validation prefix, a 65,536-token effective batch, BF16, PyTorch SDPA, and a fused AdamW optimizer when available. Each run emits its exact config, step metrics, and final result.

## Data pipeline

Common Crawl WET records pass through language identification, Gopher-style rules, a learned quality classifier, NSFW/toxicity classifiers, PII masking, exact deduplication, and tokenization.

## Alignment

Standard GRPO, DR-GRPO, and a clipped off-policy variant share the same model, prompts, train/validation splits, and rollout budget. Four seeds are reported; collapsed seeds remain in the aggregate rather than being filtered out.
