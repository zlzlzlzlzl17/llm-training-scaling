# Tests

Original correctness tests are grouped by capability:

- `model/`: Transformer, optimizer, tokenizer, BPE, data loading, serialization.
- `systems/`: attention, distributed wrappers, FSDP, optimizer sharding.
- `data_pipeline/`: extraction, language/quality filters, PII, toxicity, deduplication.
- `alignment/`: GRPO and supporting assignment interfaces.

Small fixtures remain in Git. Large alignment model weights are kept locally but ignored; the GRPO snapshot tests do not require committing those weights. Distributed tests require an appropriate CUDA/NCCL environment.
