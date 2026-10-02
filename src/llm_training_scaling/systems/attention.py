from __future__ import annotations

import math

import torch


def pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Standard scaled dot-product attention.

    Expected shapes:
        q: [..., n_queries, d]
        k: [..., n_keys, d]
        v: [..., n_keys, d]

    Returns:
        [..., n_queries, d]
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q and k must have the same embedding dimension, "
            f"got {q.shape[-1]} and {k.shape[-1]}"
        )

    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"k and v must have the same sequence length, "
            f"got {k.shape[-2]} and {v.shape[-2]}"
        )

    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]

        query_indices = torch.arange(
            n_queries,
            device=q.device,
        )[:, None]

        key_indices = torch.arange(
            n_keys,
            device=q.device,
        )[None, :]

        causal_mask = key_indices > query_indices
        scores = scores.masked_fill(causal_mask, -1e6)

    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)
