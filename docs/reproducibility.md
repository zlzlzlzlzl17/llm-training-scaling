# Reproducibility

## Environment

- Python 3.12
- CUDA-capable PyTorch for GPU experiments
- NCCL for distributed benchmarks
- Optional Triton/vLLM dependencies for kernel and alignment experiments

Install the relevant extras from `pyproject.toml`. Exact remote environments originally used NVIDIA A100 80 GB GPUs for the scaling sweep and two-GPU configurations for distributed systems tests.

The optional Modal data pipeline requires an explicit `SUNET_ID` environment variable. This repository never supplies account identifiers or credentials by default.

## Data

Datasets and checkpoints are intentionally excluded. Scripts accept explicit paths, while completed run configs under `experiments/` preserve the original inputs and hyperparameters. Original absolute paths are historical metadata and are not required to match a new machine.

## Results

Curated CSV/JSON/JSONL files are committed. Large Nsight traces should be distributed through a release or external artifact store, not ordinary Git history.
