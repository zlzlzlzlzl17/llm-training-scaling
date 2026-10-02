# Results

## End-to-end language-model training

A 14,761,344-parameter Transformer was trained for 5,000 optimizer steps on TinyStories with BF16 and a 4,096-token effective batch. It reached a final validation loss of 2.2002 (perplexity 9.03); the best checkpoint-equivalent validation point was step 4,500 at loss 2.1792 (perplexity 8.84). Steady-state training throughput was generally 62k–68k tokens/s.

The preceding data path trained a 10,000-token BPE vocabulary with 9,743 merges and encoded 541.2M training tokens at approximately 303.7k tokens/s.

## GPU systems

- Triton FlashAttention averaged 3.48× forward and 1.46× end-to-end speedup across 76 successful comparisons.
- Overlapped DDP improved the XL-model step from 438.0 ms to 360.3 ms.
- Sharded AdamW reduced post-step allocated memory by 24.9% and halved optimizer-state memory.
- FSDP forward prefetch improved step time by 6.6% with nearly unchanged peak memory.

## Scaling

| Compute budget | Best run | Parameters | Train tokens | Validation loss |
|---:|---|---:|---:|---:|
| 1e17 | c1e17_m20 | 19.3M | 864.7M | 3.8644 |
| 3e17 | c3e17_m50 | 49.6M | 1.008B | 3.6101 |
| 8e17 | c8e17_m50 | 49.6M | 2.689B | 3.4188 |
| 2e18 | c2e18_m120 | 117.0M | 2.849B | 3.2555 |

## Data pipeline

The pipeline processed 408,590 Common Crawl conversion records. It retained 14,768 documents after filtering and produced 29,684,588 GPT-2 tokens from 14,486 documents after deduplication.

## Alignment

Standard GRPO reached approximately 44% mean final validation accuracy over four seeds without collapse. DR-GRPO and clipped off-policy training each collapsed on two seeds, illustrating the sensitivity of reasoning-RL training to normalization and off-policy updates.
