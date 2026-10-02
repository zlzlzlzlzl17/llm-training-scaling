from __future__ import annotations

import math

import torch


def flash_attention_backward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute attention probabilities and calculate dQ, dK, dV."""
    input_dtype = q.dtype
    scale = 1.0 / math.sqrt(q.shape[-1])

    q_fp32 = q.float()
    k_fp32 = k.float()
    v_fp32 = v.float()
    output_fp32 = output.float()
    grad_output_fp32 = grad_output.float()

    scores = torch.matmul(
        q_fp32,
        k_fp32.transpose(-2, -1),
    ) * scale

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

        scores = scores.masked_fill(
            key_indices > query_indices,
            -1e6,
        )

    # P_ij = exp(S_ij - L_i)
    probabilities = torch.exp(
        scores - logsumexp.float().unsqueeze(-1)
    )

    # D_i = rowsum(O_i * dO_i)
    d_vector = torch.sum(
        output_fp32 * grad_output_fp32,
        dim=-1,
        keepdim=True,
    )

    grad_v = torch.matmul(
        probabilities.transpose(-2, -1),
        grad_output_fp32,
    )

    grad_p = torch.matmul(
        grad_output_fp32,
        v_fp32.transpose(-2, -1),
    )

    grad_s = probabilities * (grad_p - d_vector)

    grad_q = torch.matmul(grad_s, k_fp32) * scale
    grad_k = torch.matmul(
        grad_s.transpose(-2, -1),
        q_fp32,
    ) * scale

    return (
        grad_q.to(input_dtype),
        grad_k.to(k.dtype),
        grad_v.to(v.dtype),
    )


class FlashAttentionPytorch(torch.autograd.Function):
    """Tiled FlashAttention-2 reference implemented with PyTorch."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError(
                "Expected q, k, and v with shape "
                "[batch, sequence_length, d]"
            )

        batch_size, n_queries, d = q.shape
        n_keys = k.shape[-2]

        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have identical shapes, "
                f"got {k.shape} and {v.shape}"
            )

        if k.shape[-1] != d:
            raise ValueError(
                "q and k must have the same embedding dimension"
            )

        scale = 1.0 / math.sqrt(d)

        # Reference implementation: fixed tiles are sufficient.
        query_tile_size = 32
        key_tile_size = 32

        output = torch.empty_like(q)
        logsumexp = torch.empty(
            batch_size,
            n_queries,
            device=q.device,
            dtype=torch.float32,
        )

        for query_start in range(
            0,
            n_queries,
            query_tile_size,
        ):
            query_end = min(
                query_start + query_tile_size,
                n_queries,
            )

            q_tile = q[
                :,
                query_start:query_end,
                :,
            ].float()

            tile_rows = query_end - query_start

            # Running values for online softmax.
            running_max = torch.full(
                (batch_size, tile_rows),
                float("-inf"),
                device=q.device,
                dtype=torch.float32,
            )

            running_sum = torch.zeros(
                batch_size,
                tile_rows,
                device=q.device,
                dtype=torch.float32,
            )

            output_accumulator = torch.zeros(
                batch_size,
                tile_rows,
                d,
                device=q.device,
                dtype=torch.float32,
            )

            for key_start in range(
                0,
                n_keys,
                key_tile_size,
            ):
                key_end = min(
                    key_start + key_tile_size,
                    n_keys,
                )

                # For causal attention, all later key tiles are masked.
                if is_causal and key_start >= query_end:
                    break

                k_tile = k[
                    :,
                    key_start:key_end,
                    :,
                ].float()

                v_tile = v[
                    :,
                    key_start:key_end,
                    :,
                ].float()

                scores = torch.matmul(
                    q_tile,
                    k_tile.transpose(-2, -1),
                ) * scale

                if is_causal:
                    query_indices = torch.arange(
                        query_start,
                        query_end,
                        device=q.device,
                    )[:, None]

                    key_indices = torch.arange(
                        key_start,
                        key_end,
                        device=q.device,
                    )[None, :]

                    scores = scores.masked_fill(
                        (
                            key_indices
                            > query_indices
                        ).unsqueeze(0),
                        -1e6,
                    )

                tile_max = scores.max(dim=-1).values
                new_max = torch.maximum(
                    running_max,
                    tile_max,
                )

                # Rescale previous partial sums when the maximum changes.
                correction = torch.exp(
                    running_max - new_max
                )

                probabilities_unnormalized = torch.exp(
                    scores - new_max.unsqueeze(-1)
                )

                running_sum = (
                    correction * running_sum
                    + probabilities_unnormalized.sum(
                        dim=-1
                    )
                )

                output_accumulator = (
                    correction.unsqueeze(-1)
                    * output_accumulator
                    + torch.matmul(
                        probabilities_unnormalized,
                        v_tile,
                    )
                )

                running_max = new_max

            output_tile = (
                output_accumulator
                / running_sum.unsqueeze(-1)
            )

            output[
                :,
                query_start:query_end,
                :,
            ] = output_tile.to(q.dtype)

            logsumexp[
                :,
                query_start:query_end,
            ] = running_max + torch.log(running_sum)

        # Exactly one saved tensor has shape [batch, n_queries]: L.
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

        grad_q, grad_k, grad_v = (
            flash_attention_backward_pytorch(
                q=q,
                k=k,
                v=v,
                output=output,
                grad_output=grad_output,
                logsumexp=logsumexp,
                is_causal=ctx.is_causal,
            )
        )

        return grad_q, grad_k, grad_v, None
