from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from llm_training_scaling.model.bpe import train_bpe
from llm_training_scaling.model.bpe import train_bpe
from llm_training_scaling.model.tokenizer import Tokenizer
from llm_training_scaling.model.transformer import (
    CausalMultiHeadSelfAttention,
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from llm_training_scaling.model.nn_utils import cross_entropy
from llm_training_scaling.model.optimizer import (
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from llm_training_scaling.model.data import get_batch
from llm_training_scaling.model.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation
    of a batched input.
    """

    linear = Linear(
        in_features=d_in,
        out_features=d_out,
        device=weights.device,
        dtype=weights.dtype,
    )

    linear.load_state_dict(
        {
            "weight": weights,
        }
    )

    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings
    for a batch of token IDs.
    """

    embedding = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=weights.device,
        dtype=weights.dtype,
    )

    embedding.load_state_dict(
        {
            "weight": weights,
        }
    )

    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Run SwiGLU using the supplied reference weights.
    """

    swiglu = SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
        device=w1_weight.device,
        dtype=w1_weight.dtype,
    )

    swiglu.load_state_dict(
        {
            "w1.weight": w1_weight,
            "w2.weight": w2_weight,
            "w3.weight": w3_weight,
        }
    )

    return swiglu(in_features)

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Run scaled dot-product attention.
    """

    return scaled_dot_product_attention(
        queries=Q,
        keys=K,
        values=V,
        mask=mask,
    )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[
        Tensor,
        " ... sequence_length d_model",
    ],
) -> Float[
    Tensor,
    " ... sequence_length d_model",
]:
    """
    Run causal multi-head self-attention without RoPE.
    """

    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    attention.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )

    return attention(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[
        Tensor,
        " ... sequence_length d_model",
    ],
    token_positions: Int[
        Tensor,
        " ... sequence_length",
    ]
    | None = None,
) -> Float[
    Tensor,
    " ... sequence_length d_model",
]:
    """
    Run causal multi-head self-attention with RoPE.
    """

    attention = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        theta=theta,
        max_seq_len=max_seq_len,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    attention.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )

    return attention(
        in_features,
        token_positions=token_positions,
    )


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[
        Tensor,
        " ... sequence_length d_k",
    ],
    token_positions: Int[
        Tensor,
        " ... sequence_length",
    ],
) -> Float[
    Tensor,
    " ... sequence_length d_k",
]:
    """
    Run RoPE for a given query or key tensor.
    """

    rope = RotaryPositionalEmbedding(
        theta=theta,
        d_k=d_k,
        max_seq_len=max_seq_len,
        device=in_query_or_key.device,
    )

    return rope(
        in_query_or_key,
        token_positions,
    )

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[
        Tensor,
        " batch sequence_length d_model",
    ],
) -> Float[
    Tensor,
    " batch sequence_length d_model",
]:
    """
    Run one pre-norm Transformer block using the supplied
    reference weights.
    """

    reference_weight = weights[
        "attn.q_proj.weight"
    ]

    block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        device=reference_weight.device,
        dtype=reference_weight.dtype,
    )

    block.load_state_dict(weights)

    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[
        Tensor,
        " batch_size sequence_length",
    ],
) -> Float[
    Tensor,
    " batch_size sequence_length vocab_size",
]:
    """
    Run a complete Transformer language model using the
    supplied reference weights.
    """

    reference_weight = weights[
        "token_embeddings.weight"
    ]

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=reference_weight.device,
        dtype=reference_weight.dtype,
    )

    model.load_state_dict(weights)

    return model(in_indices)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an RMSNorm affine transform,
    return the output of running RMSNorm on the input features.
    """

    rmsnorm = RMSNorm(
        d_model=d_model,
        eps=eps,
        device=weights.device,
        dtype=weights.dtype,
    )

    rmsnorm.load_state_dict(
        {
            "weight": weights,
        }
    )

    return rmsnorm(in_features)


def run_silu(
    in_features: Float[Tensor, " ..."],
) -> Float[Tensor, " ..."]:
    """
    Apply SiLU element-wise.
    """

    return silu(in_features)

def run_get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language-modeling inputs and next-token targets.
    """

    return get_batch(
        dataset=dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )


def run_softmax(
    in_features: Float[Tensor, " ..."],
    dim: int,
) -> Float[Tensor, " ..."]:
    """
    Apply softmax along the requested dimension.
    """

    return softmax(
        in_features,
        dim=dim,
    )


def run_cross_entropy(
    inputs: Float[
        Tensor,
        " batch_size vocab_size",
    ],
    targets: Int[
        Tensor,
        " batch_size",
    ],
) -> Float[Tensor, ""]:
    """
    Compute the average cross-entropy loss across examples.
    """

    return cross_entropy(
        inputs=inputs,
        targets=targets,
    )


def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clip the global L2 norm of parameter gradients in place.
    """

    gradient_clipping(
        parameters=parameters,
        max_l2_norm=max_l2_norm,
    )


def get_adamw_cls() -> Any:
    """
    Return the custom AdamW optimizer class.
    """

    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Return the learning rate at iteration `it`.
    """

    return get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=out,
    )


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    return load_checkpoint(
        src=src,
        model=model,
        optimizer=optimizer,
    )


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=special_tokens,
)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return train_bpe(
    input_path=input_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)
