from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from llm_training_scaling.systems.flash_attention import (
    flash_attention_backward_pytorch,
)


flash_attention_backward_compiled = torch.compile(
    flash_attention_backward_pytorch,
    fullgraph=True,
    dynamic=True,
)


@triton.jit
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    n_queries,
    n_keys,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    One program handles:
        one batch element
        one query tile

    It loops over all K/V tiles and performs online softmax.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    query_start = query_tile_index * Q_TILE_SIZE
    query_offsets = (
        query_start + tl.arange(0, Q_TILE_SIZE)
    )

    # Offset pointers to the current batch.
    q_batch_ptr = q_ptr + batch_index * stride_qb
    k_batch_ptr = k_ptr + batch_index * stride_kb
    v_batch_ptr = v_ptr + batch_index * stride_vb
    o_batch_ptr = o_ptr + batch_index * stride_ob
    l_batch_ptr = l_ptr + batch_index * stride_lb

    q_block_ptr = tl.make_block_ptr(
        base=q_batch_ptr,
        shape=(n_queries, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=o_batch_ptr,
        shape=(n_queries, D),
        strides=(stride_oq, stride_od),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    l_block_ptr = tl.make_block_ptr(
        base=l_batch_ptr,
        shape=(n_queries,),
        strides=(stride_lq,),
        offsets=(query_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(
        q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # Online-softmax state, always accumulated in FP32.
    running_max = tl.full(
        (Q_TILE_SIZE,),
        -float("inf"),
        dtype=tl.float32,
    )
    running_sum = tl.zeros(
        (Q_TILE_SIZE,),
        dtype=tl.float32,
    )
    output_accumulator = tl.zeros(
        (Q_TILE_SIZE, D),
        dtype=tl.float32,
    )

    for key_start in range(
        0,
        n_keys,
        K_TILE_SIZE,
    ):
        key_offsets = (
            key_start + tl.arange(0, K_TILE_SIZE)
        )

        k_block_ptr = tl.make_block_ptr(
            base=k_batch_ptr,
            shape=(n_keys, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        v_block_ptr = tl.make_block_ptr(
            base=v_batch_ptr,
            shape=(n_keys, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k = tl.load(
            k_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        v = tl.load(
            v_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        # [Q_TILE_SIZE, D] @ [D, K_TILE_SIZE]
        scores = tl.dot(q, tl.trans(k)) * scale

        # Mask keys outside the real sequence.
        valid_keys = key_offsets < n_keys
        scores = tl.where(
            valid_keys[None, :],
            scores,
            -1.0e6,
        )

        if IS_CAUSAL:
            causal_mask = (
                query_offsets[:, None]
                >= key_offsets[None, :]
            )
            scores = tl.where(
                causal_mask,
                scores,
                -1.0e6,
            )

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(
            running_max,
            tile_max,
        )

        correction = tl.exp(
            running_max - new_max
        )

        probabilities = tl.exp(
            scores - new_max[:, None]
        )

        new_sum = (
            correction * running_sum
            + tl.sum(probabilities, axis=1)
        )

        # Previous accumulator must be rescaled whenever m changes.
        output_accumulator *= correction[:, None]

        # Cast probabilities to V's dtype before tensor-core matmul.
        probabilities_compute = probabilities.to(v.dtype)

        output_accumulator = tl.dot(
            probabilities_compute,
            v,
            acc=output_accumulator,
        )

        running_max = new_max
        running_sum = new_sum

    output = (
        output_accumulator
        / running_sum[:, None]
    )
    logsumexp = (
        running_max + tl.log(running_sum)
    )

    # Accumulate and normalize in FP32, then cast back to the
    # output tensor dtype before storing through the block pointer.
    tl.store(
        o_block_ptr,
        output.to(q.dtype),
        boundary_check=(0, 1),
    )
    tl.store(
        l_block_ptr,
        logsumexp,
        boundary_check=(0,),
    )


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 with a fused Triton forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not q.is_cuda:
            raise ValueError(
                "Triton FlashAttention requires CUDA tensors"
            )

        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError(
                "Expected q, k, v with shape [batch, sequence, d]"
            )

        if k.shape != v.shape:
            raise ValueError(
                f"k and v shapes differ: {k.shape} and {v.shape}"
            )

        batch_size, n_queries, d = q.shape
        n_keys = k.shape[-2]

        if k.shape[-1] != d:
            raise ValueError(
                "q and k must have the same embedding dimension"
            )

        if d < 16:
            raise ValueError(
                f"D must be at least 16, got {d}"
            )

        if d & (d - 1):
            raise ValueError(
                f"D must be a power of two, got {d}"
            )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        output = torch.empty_like(q)
        logsumexp = torch.empty(
            batch_size,
            n_queries,
            device=q.device,
            dtype=torch.float32,
        )

        query_tile_size = 32
        key_tile_size = 32

        grid = (
            triton.cdiv(
                n_queries,
                query_tile_size,
            ),
            batch_size,
        )

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            logsumexp.stride(0),
            logsumexp.stride(1),
            n_queries,
            n_keys,
            1.0 / math.sqrt(d),
            D=d,
            Q_TILE_SIZE=query_tile_size,
            K_TILE_SIZE=key_tile_size,
            IS_CAUSAL=bool(is_causal),
            num_warps=4,
            num_stages=3,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            logsumexp,
        )
        ctx.is_causal = bool(is_causal)

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        q, k, v, output, logsumexp = ctx.saved_tensors

        # The assignment permits the backward equations to be written
        # in PyTorch. It recomputes P instead of saving the N x N matrix.
        grad_q, grad_k, grad_v = (
            flash_attention_backward_compiled(
                q=q,
                k=k,
                v=v,
                output=output,
                grad_output=grad_output.contiguous(),
                logsumexp=logsumexp,
                is_causal=ctx.is_causal,
            )
        )

        return grad_q, grad_k, grad_v, None
