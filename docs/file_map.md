# Canonical file map

The dated snapshot remains unchanged. The showcase repository contains copied canonical versions under capability-oriented names.

| New location | Snapshot source |
|---|---|
| `src/llm_training_scaling/model/` | Assignment 1 `cs336_basics/` (byte-identical to the later Assignment 3 working copy) |
| `src/llm_training_scaling/systems/` | `assignment2/repo/cs336_systems/` |
| `src/llm_training_scaling/systems/benchmark_model.py` | Assignment 2's compatible `cs336-basics/model.py` |
| `src/llm_training_scaling/data_pipeline/` | `assignment4-data/cs336_data/` plus its compatible training backend |
| `src/llm_training_scaling/alignment/` | `assignment5/repo/cs336_alignment/` |
| `scripts/training/` | Assignment 3 tokenizer, training, scaling, and generation drivers |
| `scripts/training/analyze_training.py` | Assignment 1 training-log analysis utility |
| `scripts/benchmark/` | Assignment 2 benchmark drivers |
| `scripts/data_pipeline/` | Assignment 4 download/filter/dedup/tokenization/training drivers |
| `scripts/alignment/` | Assignment 5 prompting, GRPO, safety, and multi-seed drivers |
| `experiments/systems/` | Assignment 2 result CSVs; profiler traces are local-only |
| `experiments/training/` | Assignment 1 BPE, tokenization, TinyStories config/log/summary/sample |
| `experiments/scaling/` | Assignment 3 calibration and iso-FLOP run artifacts |
| `experiments/data_pipeline/` | Assignment 4 aggregate filtering/tokenization results |
| `experiments/alignment/` | Assignment 5 compact per-run metrics and multi-seed summaries |

The Assignment 3 `assignment3_scaling_stage1` deployment copy, backup files, caches, raw data, checkpoints, and duplicate archives were not copied into the showcase tree. They remain recoverable from `remote_snapshot_20261002`.
