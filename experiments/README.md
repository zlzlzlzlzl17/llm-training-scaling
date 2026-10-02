# Experiments

This directory contains compact, auditable experiment artifacts rather than datasets or checkpoints.

- `systems/`: benchmark CSVs and local-only Nsight traces.
- `training/`: TinyStories BPE artifacts, tokenization metadata, run config/log/summary, and generated sample.
- `scaling/`: exact configs, step metrics, and final results for 22 completed runs.
- `data_pipeline/`: aggregate Common Crawl filtering and tokenization statistics.
- `alignment/`: prompting summary, per-run training metrics, multi-seed summary, and validation curves.

Large raw data, model weights, full rollouts, and web-text inspection samples remain in the dated local snapshot described in `docs/provenance.md`.
