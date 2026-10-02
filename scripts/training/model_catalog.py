from __future__ import annotations

MODELS = [
    ("M20", 6, 512, 8, 1408, 64, 2),
    ("M50", 10, 640, 10, 1728, 32, 4),
    ("M120", 12, 896, 14, 2432, 16, 8),
    ("M300", 15, 1280, 20, 3456, 8, 16),
    ("M500", 16, 1600, 25, 4288, 4, 32),
    ("M800", 16, 2048, 32, 5504, 2, 64),
]

VOCAB_SIZE = 32_000
CONTEXT_LENGTH = 512

print(
    "name,layers,d_model,heads,d_ff,head_dim,"
    "micro_batch,grad_accum,global_batch_tokens,"
    "non_embedding_params,total_params,approx_12Ld2"
)

for name, layers, d_model, heads, d_ff, micro_batch, grad_accum in MODELS:
    head_dim = d_model // heads
    non_embedding = (
        layers * (
            4 * d_model * d_model
            + 3 * d_model * d_ff
            + 2 * d_model
        )
        + d_model
    )
    total = non_embedding + 2 * VOCAB_SIZE * d_model
    approximate = 12 * layers * d_model * d_model
    global_batch_tokens = (
        micro_batch * grad_accum * CONTEXT_LENGTH
    )
    print(
        f"{name},{layers},{d_model},{heads},{d_ff},{head_dim},"
        f"{micro_batch},{grad_accum},{global_batch_tokens},"
        f"{non_embedding},{total},{approximate}"
    )
