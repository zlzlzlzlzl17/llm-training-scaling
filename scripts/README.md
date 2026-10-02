# Entry points

Run commands from the repository root after installing the package in editable mode.

- `training/`: tokenizer, binary-corpus preparation, language-model training, generation, and scaling sweeps.
- `benchmark/`: attention, FlashAttention, collective communication, DDP, optimizer sharding, and FSDP.
- `data_pipeline/`: WET download/filtering, deduplication, tokenization, classifier training, and filtered-corpus LM training.
- `alignment/`: prompting evaluation, GRPO training, failure inspection, safety evaluation, and shell launchers for the completed multi-seed comparison.

Use `python <script> --help` for script-specific arguments. GPU/distributed entry points require the corresponding optional dependencies and hardware.
