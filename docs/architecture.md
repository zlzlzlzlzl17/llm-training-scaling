# Architecture

The repository separates reusable implementations from executable experiments.

```text
scripts -> src package -> experiment artifacts -> figures
```

The `model` package supplies the base Transformer stack. `systems` adds optimized kernels and distributed wrappers around that model. `data_pipeline` prepares training text. `alignment` implements reward computation and policy-gradient training around a Transformers/vLLM policy.

Assignment-specific folder names are intentionally absent from the API. Provenance is retained in `provenance.md` and in each experiment's machine-readable configuration.
