from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """
    不含 bias 的线性变换。

    输入形状：
        (..., in_features)

    权重形状：
        (out_features, in_features)

    输出形状：
        (..., out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 按照作业要求存储 W，而不是 W^T。
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        W ~ N(0, sigma^2)，其中：

            sigma^2 = 2 / (d_in + d_out)

        并截断到 [-3 sigma, 3 sigma]。
        """

        std = math.sqrt(
            2.0 / (self.in_features + self.out_features)
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        PyTorch 把最后一维视为 feature dimension，因此计算：

            x @ W^T

        x:
            (..., in_features)

        self.weight.T:
            (in_features, out_features)

        output:
            (..., out_features)
        """

        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """
    Token embedding lookup。

    输入：
        任意形状的整数 token IDs，例如
        (batch_size, sequence_length)

    权重：
        (vocab_size, d_model)

    输出：
        输入形状后追加 d_model，例如
        (batch_size, sequence_length, d_model)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Embedding 权重：

            W ~ N(0, 1)

        并截断到 [-3, 3]。
        """

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 token ID 直接索引 embedding matrix。

        例如：
            token_ids.shape == (batch_size, sequence_length)

        则：
            output.shape ==
            (batch_size, sequence_length, embedding_dim)
        """

        return self.weight[token_ids]
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    输入形状：
        (..., d_model)

    输出形状：
        (..., d_model)

    RMSNorm 不减去均值，只根据最后一维的均方根
    对输入进行缩放。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )

        if eps <= 0:
            raise ValueError(
                f"eps must be positive, got {eps}"
            )

        self.d_model = d_model
        self.eps = eps

        # 可学习 gain 参数 g。
        # 每个 hidden dimension 对应一个缩放值。
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        """

        if x.shape[-1] != self.d_model:
            raise ValueError(
                "The final input dimension must equal d_model: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )

        # 保存输入 dtype，例如 float16、bfloat16 或 float32。
        input_dtype = x.dtype

        # 作业要求计算平方前转换为 float32，
        # 避免低精度数据平方时溢出。
        x_float = x.to(torch.float32)

        mean_square = torch.mean(
            x_float * x_float,
            dim=-1,
            keepdim=True,
        )

        inverse_rms = torch.rsqrt(
            mean_square + self.eps
        )

        normalized = x_float * inverse_rms

        # weight 也转换为 float32 参与计算，
        # 最后再恢复输入 dtype。
        result = normalized * self.weight.to(torch.float32)

        return result.to(input_dtype)

def silu(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    SiLU，也称为 Swish。

    SiLU(x) = x * sigmoid(x)

    输入与输出形状完全相同。
    """

    return x * torch.sigmoid(x)


def get_default_d_ff(
    d_model: int,
    multiple_of: int = 64,
) -> int:
    """
    根据 d_model 计算默认的 SwiGLU hidden dimension。

    d_ff 约为：

        (8 / 3) * d_model

    并向上取整到 multiple_of 的倍数。
    """

    if d_model <= 0:
        raise ValueError(
            f"d_model must be positive, got {d_model}"
        )

    if multiple_of <= 0:
        raise ValueError(
            f"multiple_of must be positive, got {multiple_of}"
        )

    approximate_d_ff = 8 * d_model / 3

    return (
        math.ceil(approximate_d_ff / multiple_of)
        * multiple_of
    )


class SwiGLU(nn.Module):
    """
    Transformer 的 position-wise feed-forward network。

    计算：

        gate = SiLU(W1 x)
        value = W3 x
        hidden = gate * value
        output = W2 hidden

    输入形状：
        (..., d_model)

    中间形状：
        (..., d_ff)

    输出形状：
        (..., d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )

        if d_ff is None:
            d_ff = get_default_d_ff(d_model)

        if d_ff <= 0:
            raise ValueError(
                f"d_ff must be positive, got {d_ff}"
            )

        self.d_model = d_model
        self.d_ff = d_ff

        # Gate projection:
        # (..., d_model) -> (..., d_ff)
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        # Down projection:
        # (..., d_ff) -> (..., d_model)
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # Value/up projection:
        # (..., d_model) -> (..., d_ff)
        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        运行 SwiGLU feed-forward network。
        """

        if x.shape[-1] != self.d_model:
            raise ValueError(
                "The final input dimension must equal d_model: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )

        gate = silu(self.w1(x))
        value = self.w3(x)
        hidden = gate * value

        return self.w2(hidden)

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding（RoPE）。

    输入形状：
        (..., sequence_length, d_k)

    token_positions 形状：
        (..., sequence_length)

    输出形状：
        (..., sequence_length, d_k)

    RoPE 没有可学习参数。cos/sin 被保存为 buffer，
    会随着 module.to(device) 一起移动。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if theta <= 0:
            raise ValueError(
                f"theta must be positive, got {theta}"
            )

        if d_k <= 0:
            raise ValueError(
                f"d_k must be positive, got {d_k}"
            )

        if d_k % 2 != 0:
            raise ValueError(
                f"d_k must be even for RoPE, got {d_k}"
            )

        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be positive, "
                f"got {max_seq_len}"
            )

        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 对应每一对 embedding dimensions：
        #
        # k = 0, 1, ..., d_k/2 - 1
        #
        # frequency_k = theta^(-2k/d_k)
        dimension_indices = torch.arange(
            0,
            d_k,
            2,
            dtype=torch.float32,
            device=device,
        )

        inverse_frequencies = torch.pow(
            torch.tensor(
                self.theta,
                dtype=torch.float32,
                device=device,
            ),
            -dimension_indices / d_k,
        )

        # positions:
        # (max_seq_len,)
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
            device=device,
        )

        # angles:
        # (max_seq_len, d_k / 2)
        angles = positions[:, None] * inverse_frequencies[None, :]

        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)

        # RoPE 的 cos/sin 是固定值，不是模型参数。
        self.register_buffer(
            "cos_cached",
            cos_cached,
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            sin_cached,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 x 的最后一维相邻元素进行成对旋转。

        对每一对 (x_even, x_odd)：

            output_even = x_even * cos - x_odd * sin
            output_odd  = x_even * sin + x_odd * cos
        """

        if x.shape[-1] != self.d_k:
            raise ValueError(
                "The final dimension of x must equal d_k: "
                f"expected {self.d_k}, got {x.shape[-1]}"
            )

        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions sequence length must match x: "
                f"got {token_positions.shape[-1]} and "
                f"{x.shape[-2]}"
            )

        if token_positions.numel() == 0:
            return x.clone()

        positions = token_positions.to(
            device=self.cos_cached.device,
            dtype=torch.long,
        )

        minimum_position = int(positions.min().item())
        maximum_position = int(positions.max().item())

        if minimum_position < 0:
            raise ValueError(
                "token_positions cannot contain negative values."
            )

        if maximum_position >= self.max_seq_len:
            raise ValueError(
                "token position exceeds max_seq_len: "
                f"maximum position is {maximum_position}, "
                f"but max_seq_len is {self.max_seq_len}"
            )

        # 得到：
        #
        # token_positions shape:
        #     (..., sequence_length)
        #
        # cos/sin shape:
        #     (..., sequence_length, d_k / 2)
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]

        # 例如：
        #
        # x shape:
        #     (batch, heads, seq, d_k)
        #
        # token_positions shape:
        #     (batch, seq)
        #
        # 此时 cos 为 (batch, seq, d_k/2)，需要增加 head 维：
        #     (batch, 1, seq, d_k/2)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 使用 float32 计算三角函数乘法更稳定，
        # 最终恢复输入 dtype。
        input_dtype = x.dtype
        x_float = x.to(torch.float32)

        cos = cos.to(
            device=x.device,
            dtype=torch.float32,
        )
        sin = sin.to(
            device=x.device,
            dtype=torch.float32,
        )

        # 相邻维度组成旋转对：
        #
        # (0, 1), (2, 3), (4, 5), ...
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]

        rotated_even = (
            x_even * cos
            - x_odd * sin
        )

        rotated_odd = (
            x_even * sin
            + x_odd * cos
        )

        # (..., seq, d_k/2, 2)
        rotated_pairs = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        # 恢复为 (..., seq, d_k)。
        output = rotated_pairs.flatten(
            start_dim=-2,
        )

        return output.to(input_dtype)

def softmax(
    x: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Numerically stable softmax。

    输入：
        任意形状 tensor

    dim：
        执行 softmax 的维度

    输出：
        与输入形状相同；沿 dim 维的元素和为 1。
    """

    if x.ndim == 0:
        raise ValueError(
            "softmax expects a tensor with at least one dimension."
        )

    if dim < -x.ndim or dim >= x.ndim:
        raise IndexError(
            f"dim={dim} is invalid for tensor with {x.ndim} dimensions."
        )

    # Softmax 对整体平移不敏感：
    #
    # softmax(x) = softmax(x - max(x))
    #
    # 减去最大值后，最大的指数输入为 0，
    # 避免 exp(large_number) 溢出。
    maximum = torch.amax(
        x,
        dim=dim,
        keepdim=True,
    )

    shifted = x - maximum
    exponentials = torch.exp(shifted)

    denominator = torch.sum(
        exponentials,
        dim=dim,
        keepdim=True,
    )

    return exponentials / denominator


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention。

    queries:
        (..., num_queries, d_k)

    keys:
        (..., num_keys, d_k)

    values:
        (..., num_keys, d_v)

    mask:
        (..., num_queries, num_keys)

        True  = 允许 query 关注该 key
        False = 屏蔽该位置

    输出：
        (..., num_queries, d_v)
    """

    if queries.ndim < 2:
        raise ValueError(
            "queries must have at least two dimensions."
        )

    if keys.ndim < 2:
        raise ValueError(
            "keys must have at least two dimensions."
        )

    if values.ndim < 2:
        raise ValueError(
            "values must have at least two dimensions."
        )

    d_k = queries.shape[-1]

    if d_k <= 0:
        raise ValueError(
            "The query/key dimension d_k must be positive."
        )

    if keys.shape[-1] != d_k:
        raise ValueError(
            "queries and keys must have the same final dimension: "
            f"got {d_k} and {keys.shape[-1]}."
        )

    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            "keys and values must contain the same number of "
            f"positions: got {keys.shape[-2]} and "
            f"{values.shape[-2]}."
        )

    # queries:
    # (..., num_queries, d_k)
    #
    # keys.transpose(-2, -1):
    # (..., d_k, num_keys)
    #
    # scores:
    # (..., num_queries, num_keys)
    scores = torch.matmul(
        queries,
        keys.transpose(-2, -1),
    )

    scores = scores / math.sqrt(d_k)

    boolean_mask: torch.Tensor | None = None

    if mask is not None:
        boolean_mask = mask.to(
            device=scores.device,
            dtype=torch.bool,
        )

        # False 位置在 softmax 前设置成 -inf。
        #
        # exp(-inf) = 0
        scores = scores.masked_fill(
            ~boolean_mask,
            float("-inf"),
        )

    # 对 key dimension 做 softmax。
    attention_weights = softmax(
        scores,
        dim=-1,
    )

    if boolean_mask is not None:
        # 当某一整行 mask 全为 False 时，softmax 内部可能产生
        # NaN。这里明确把所有被屏蔽位置设为 0，使这种情况也能
        # 返回全零 attention weights。
        attention_weights = torch.where(
            boolean_mask,
            attention_weights,
            torch.zeros_like(attention_weights),
        )

    # attention_weights:
    # (..., num_queries, num_keys)
    #
    # values:
    # (..., num_keys, d_v)
    #
    # output:
    # (..., num_queries, d_v)
    return torch.matmul(
        attention_weights,
        values,
    )

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention。

    输入形状：
        (..., sequence_length, d_model)

    输出形状：
        (..., sequence_length, d_model)

    每个 attention head 的维度：

        head_dim = d_model // num_heads

    当 theta 和 max_seq_len 均被提供时，对 Q、K 使用 RoPE。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"got d_model={d_model}, num_heads={num_heads}"
            )

        if (theta is None) != (max_seq_len is None):
            raise ValueError(
                "theta and max_seq_len must either both be provided "
                "or both be None."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q、K、V projection 分别用一次矩阵乘法处理全部 heads。
        #
        # (..., seq_len, d_model)
        #     ->
        # (..., seq_len, d_model)
        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # 合并所有 heads 后执行 output projection。
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # theta 与 max_seq_len 均提供时启用 RoPE。
        if theta is not None and max_seq_len is not None:
            self.rope: RotaryPositionalEmbedding | None = (
                RotaryPositionalEmbedding(
                    theta=theta,
                    d_k=self.head_dim,
                    max_seq_len=max_seq_len,
                    device=device,
                )
            )
        else:
            self.rope = None

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        将最后一个 d_model 维拆分为：

            num_heads × head_dim

        输入：
            (..., sequence_length, d_model)

        reshape 后：
            (..., sequence_length, num_heads, head_dim)

        交换维度后：
            (..., num_heads, sequence_length, head_dim)
        """

        sequence_length = x.shape[-2]

        x = x.reshape(
            *x.shape[:-2],
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(-3, -2)

    def _merge_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        合并 attention heads。

        输入：
            (..., num_heads, sequence_length, head_dim)

        输出：
            (..., sequence_length, d_model)
        """

        sequence_length = x.shape[-2]

        # (..., num_heads, seq, head_dim)
        #     ->
        # (..., seq, num_heads, head_dim)
        x = x.transpose(-3, -2)

        return x.reshape(
            *x.shape[:-3],
            sequence_length,
            self.d_model,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        执行 causal multi-head self-attention。

        token_positions:
            (..., sequence_length)

        当启用 RoPE 且 token_positions=None 时，默认使用：

            [0, 1, ..., sequence_length - 1]
        """

        if x.ndim < 2:
            raise ValueError(
                "Input must have shape "
                "(..., sequence_length, d_model)."
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                "The final input dimension must equal d_model: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )

        sequence_length = x.shape[-2]

        # ------------------------------------------------------
        # 1. 计算所有 heads 的 Q、K、V
        # ------------------------------------------------------

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # ------------------------------------------------------
        # 2. 拆分 heads
        #
        # (..., seq, d_model)
        #     ->
        # (..., heads, seq, head_dim)
        # ------------------------------------------------------

        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # ------------------------------------------------------
        # 3. RoPE 只应用于 Q 和 K
        # ------------------------------------------------------

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                token_positions = token_positions.to(
                    device=x.device,
                    dtype=torch.long,
                )

            queries = self.rope(
                queries,
                token_positions,
            )

            keys = self.rope(
                keys,
                token_positions,
            )

        # ------------------------------------------------------
        # 4. 构造 causal mask
        #
        # query i 只能关注 key j，其中 j <= i。
        #
        # 对 seq_len = 4：
        #
        # [[T, F, F, F],
        #  [T, T, F, F],
        #  [T, T, T, F],
        #  [T, T, T, T]]
        # ------------------------------------------------------

        positions = torch.arange(
            sequence_length,
            device=x.device,
        )

        causal_mask = (
            positions[:, None]
            >= positions[None, :]
        )

        # ------------------------------------------------------
        # 5. 每个 head 独立执行 attention
        #
        # head 维会被 scaled_dot_product_attention 当作
        # batch-like dimension。
        # ------------------------------------------------------

        attention_output = scaled_dot_product_attention(
            queries=queries,
            keys=keys,
            values=values,
            mask=causal_mask,
        )

        # ------------------------------------------------------
        # 6. 合并所有 heads
        # ------------------------------------------------------

        attention_output = self._merge_heads(
            attention_output
        )

        # ------------------------------------------------------
        # 7. Output projection
        # ------------------------------------------------------

        return self.output_proj(attention_output)


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block。

    计算过程：

        h = x + Attention(RMSNorm(x))
        y = h + SwiGLU(RMSNorm(h))

    输入形状：
        (..., sequence_length, d_model)

    输出形状：
        (..., sequence_length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"d_model={d_model}, num_heads={num_heads}"
            )

        if d_ff <= 0:
            raise ValueError(
                f"d_ff must be positive, got {d_ff}"
            )

        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be positive, "
                f"got {max_seq_len}"
            )

        if theta <= 0:
            raise ValueError(
                f"theta must be positive, got {theta}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 第一层 RMSNorm：在 Attention 前执行。
        self.ln1 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # Causal self-attention，并启用 RoPE。
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        # 第二层 RMSNorm：在 FFN 前执行。
        self.ln2 = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # SwiGLU feed-forward network。
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        运行一个 Transformer block。

        token_positions:
            (..., sequence_length)

        如果为 None，Attention 内部默认使用：
            [0, 1, ..., sequence_length - 1]
        """

        if x.ndim < 2:
            raise ValueError(
                "x must have shape "
                "(..., sequence_length, d_model)."
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                "The final dimension of x must equal d_model: "
                f"expected {self.d_model}, got {x.shape[-1]}"
            )

        # ------------------------------------------------------
        # Attention sub-layer
        #
        # h = x + Attention(RMSNorm(x))
        # ------------------------------------------------------

        normalized_x = self.ln1(x)

        attention_output = self.attn(
            normalized_x,
            token_positions=token_positions,
        )

        hidden = x + attention_output

        # ------------------------------------------------------
        # Feed-forward sub-layer
        #
        # output = h + SwiGLU(RMSNorm(h))
        # ------------------------------------------------------

        normalized_hidden = self.ln2(hidden)

        feed_forward_output = self.ffn(
            normalized_hidden
        )

        return hidden + feed_forward_output

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer Language Model。

    结构：

        token IDs
            ↓
        Token Embedding
            ↓
        num_layers × TransformerBlock
            ↓
        Final RMSNorm
            ↓
        LM Head
            ↓
        Next-token logits

    输入形状：
        (batch_size, sequence_length)

    输出形状：
        (batch_size, sequence_length, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(
                f"vocab_size must be positive, got {vocab_size}"
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive, "
                f"got {context_length}"
            )

        if d_model <= 0:
            raise ValueError(
                f"d_model must be positive, got {d_model}"
            )

        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"d_model={d_model}, num_heads={num_heads}"
            )

        if d_ff <= 0:
            raise ValueError(
                f"d_ff must be positive, got {d_ff}"
            )

        if rope_theta <= 0:
            raise ValueError(
                f"rope_theta must be positive, got {rope_theta}"
            )

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # ------------------------------------------------------
        # Token embedding
        #
        # (batch, sequence_length)
        #     ->
        # (batch, sequence_length, d_model)
        # ------------------------------------------------------

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # ------------------------------------------------------
        # Transformer blocks
        #
        # ModuleList 会正确注册每一层的参数，并产生类似：
        #
        # layers.0.attn.q_proj.weight
        # layers.1.attn.q_proj.weight
        # ...
        # ------------------------------------------------------

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Pre-norm architecture 在所有 block 后还需要一次
        # final RMSNorm。
        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        )

        # ------------------------------------------------------
        # Language-model head
        #
        # (..., d_model)
        #     ->
        # (..., vocab_size)
        #
        # 不包含 bias，也不与 token embedding 强制共享权重。
        # ------------------------------------------------------

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        in_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 token IDs 执行 Transformer forward pass。

        返回的是未经过 Softmax 的 logits。
        """

        if in_indices.ndim != 2:
            raise ValueError(
                "in_indices must have shape "
                "(batch_size, sequence_length), "
                f"got {tuple(in_indices.shape)}"
            )

        sequence_length = in_indices.shape[1]

        if sequence_length > self.context_length:
            raise ValueError(
                "Input sequence length exceeds context_length: "
                f"sequence_length={sequence_length}, "
                f"context_length={self.context_length}"
            )

        # Embedding 索引应为整数，并且和模型参数位于同一设备。
        in_indices = in_indices.to(
            device=self.token_embeddings.weight.device,
            dtype=torch.long,
        )

        # ------------------------------------------------------
        # 1. Token embedding
        #
        # (batch, sequence_length)
        #     ->
        # (batch, sequence_length, d_model)
        # ------------------------------------------------------

        hidden = self.token_embeddings(in_indices)

        # 所有 batch 使用相同的位置：
        #
        # [0, 1, ..., sequence_length - 1]
        token_positions = torch.arange(
            sequence_length,
            device=hidden.device,
            dtype=torch.long,
        )

        # ------------------------------------------------------
        # 2. Transformer blocks
        # ------------------------------------------------------

        for layer in self.layers:
            hidden = layer(
                hidden,
                token_positions=token_positions,
            )

        # ------------------------------------------------------
        # 3. Final RMSNorm
        # ------------------------------------------------------

        hidden = self.ln_final(hidden)

        # ------------------------------------------------------
        # 4. LM head
        #
        # 返回 logits，不执行 Softmax。
        # ------------------------------------------------------

        logits = self.lm_head(hidden)

        return logits

