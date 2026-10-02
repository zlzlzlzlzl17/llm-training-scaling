from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import regex as re


# GPT-2 风格的 pre-tokenization regex。
PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# 预先编译 regex，避免每次调用时重复编译。
PRETOKEN_PATTERN = re.compile(PAT)

# 预先创建全部 256 个单 byte token。
# BYTE_TOKENS[97] == b"a"
BYTE_TOKENS = tuple(bytes([value]) for value in range(256))


def _split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    """
    按 special token 切分文本。

    Special token 本身不会包含在返回结果中，因此：
    1. special token 不参与 merge frequency；
    2. special token 左右两边不会产生跨边界 pair。
    """
    if not special_tokens:
        return [text]

    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens cannot be empty strings.")

    # 长 special token 优先匹配，防止短 token 抢先匹配长 token 的前缀。
    tokens_for_matching = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )

    # re.escape 防止 special token 中的 |、[、] 等字符
    # 被解释成 regex 运算符。
    escaped_tokens = [
        re.escape(token)
        for token in tokens_for_matching
    ]

    special_pattern = re.compile("|".join(escaped_tokens))

    return special_pattern.split(text)


def _text_to_byte_tokens(text: str) -> tuple[bytes, ...]:
    """
    把一个 pre-token 字符串转换为初始 byte token 序列。

    例如：
        "low" -> (b"l", b"o", b"w")

        "牛" 的 UTF-8 有三个 bytes，因此会得到三个初始 token。
    """
    encoded = text.encode("utf-8")

    return tuple(
        BYTE_TOKENS[byte_value]
        for byte_value in encoded
    )


def _merge_pair(
    tokens: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """
    在一个 pre-token 内，从左到右合并指定 pair。

    例如：
        tokens = (b"a", b"a", b"a")
        pair = (b"a", b"a")

    返回：
        (b"aa", b"a")

    同一个位置不能同时参与两次 merge。
    """
    left, right = pair
    merged_token = left + right

    output: list[bytes] = []
    index = 0
    token_count = len(tokens)

    while index < token_count:
        can_merge = (
            index + 1 < token_count
            and tokens[index] == left
            and tokens[index + 1] == right
        )

        if can_merge:
            output.append(merged_token)
            index += 2
        else:
            output.append(tokens[index])
            index += 1

    return tuple(output)


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]],
]:
    """
    训练 byte-level BPE tokenizer。

    Parameters
    ----------
    input_path:
        BPE 训练文本文件路径。

    vocab_size:
        最终 vocabulary 的最大大小，包含：
        - 256 个原始 byte token；
        - special tokens；
        - merge 产生的新 token。

    special_tokens:
        加入 vocabulary 的 special tokens。
        它们是训练语料中的 hard boundaries，并且不参与 pair 统计。

    Returns
    -------
    vocab:
        token ID 到 token bytes 的映射。

    merges:
        按创建顺序排列的 BPE merge pairs。
    """

    # ==========================================================
    # 1. 处理 special tokens
    # ==========================================================

    # 去除重复 token，同时保留用户给出的顺序。
    unique_special_tokens = list(dict.fromkeys(special_tokens))

    minimum_vocab_size = 256 + len(unique_special_tokens)

    if vocab_size < minimum_vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} is too small. "
            f"At least {minimum_vocab_size} entries are required "
            "for the 256 byte tokens and the special tokens."
        )

    # ==========================================================
    # 2. 初始化 vocabulary
    # ==========================================================

    vocab: dict[int, bytes] = {
        token_id: BYTE_TOKENS[token_id]
        for token_id in range(256)
    }

    # special token 也以 UTF-8 bytes 形式保存在 vocabulary 中。
    for special_token in unique_special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    # ==========================================================
    # 3. 读取训练语料
    # ==========================================================

    text = Path(input_path).read_text(encoding="utf-8")

    # ==========================================================
    # 4. 按 special tokens 分割文本
    # ==========================================================

    text_segments = _split_on_special_tokens(
        text=text,
        special_tokens=unique_special_tokens,
    )

    # ==========================================================
    # 5. Pre-tokenization
    #
    # key:
    #   一个 pre-token 当前的 token 序列
    #
    # value:
    #   这个 pre-token 在 corpus 中的出现次数
    # ==========================================================

    pretoken_counts: dict[tuple[bytes, ...], int] = {}

    for segment in text_segments:
        for match in PRETOKEN_PATTERN.finditer(segment):
            pretoken_text = match.group(0)
            byte_tokens = _text_to_byte_tokens(pretoken_text)

            if byte_tokens:
                pretoken_counts[byte_tokens] = (
                    pretoken_counts.get(byte_tokens, 0) + 1
                )

    # ==========================================================
    # 6. 把不同 pre-token 存入列表
    #
    # words[word_id]:
    #   当前 token 序列
    #
    # word_frequencies[word_id]:
    #   该 pre-token 的语料频率
    # ==========================================================

    words: list[tuple[bytes, ...]] = list(pretoken_counts.keys())

    word_frequencies: list[int] = [
        pretoken_counts[word]
        for word in words
    ]

    # ==========================================================
    # 7. 建立初始 pair frequency 和反向索引
    #
    # pair_counts:
    #   pair -> 该 pair 在整个 corpus 中的加权频率
    #
    # pair_to_words:
    #   pair -> 包含这个 pair 的 word ID 集合
    # ==========================================================

    pair_counts: dict[tuple[bytes, bytes], int] = {}

    pair_to_words: dict[
        tuple[bytes, bytes],
        set[int],
    ] = defaultdict(set)

    for word_id, tokens in enumerate(words):
        frequency = word_frequencies[word_id]

        # 一个 pair 可能在同一个 pre-token 中出现多次。
        # pair_counts 要计算每一次出现；
        # pair_to_words 只需要保存一次 word_id。
        pairs_seen_in_word: set[tuple[bytes, bytes]] = set()

        for index in range(len(tokens) - 1):
            pair = (
                tokens[index],
                tokens[index + 1],
            )

            pair_counts[pair] = (
                pair_counts.get(pair, 0)
                + frequency
            )

            pairs_seen_in_word.add(pair)

        for pair in pairs_seen_in_word:
            pair_to_words[pair].add(word_id)

    # ==========================================================
    # 8. 反复执行 BPE merge
    # ==========================================================

    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size and pair_counts:
        # ------------------------------------------------------
        # 8.1 选择最高频 pair
        #
        # item:
        #   (pair, frequency)
        #
        # item[1]:
        #   frequency
        #
        # item[0]:
        #   pair
        #
        # 先比较 frequency；相同时比较 pair 的字典序。
        # ------------------------------------------------------

        best_pair = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

        merged_token = best_pair[0] + best_pair[1]

        # 每次 merge 产生一个新的 vocabulary token。
        vocab[len(vocab)] = merged_token

        # merge rule 必须按创建顺序保存。
        merges.append(best_pair)

        # ------------------------------------------------------
        # 8.2 只更新真正包含 best_pair 的 pre-token
        # ------------------------------------------------------

        affected_word_ids = tuple(
            pair_to_words.get(best_pair, ())
        )

        for word_id in affected_word_ids:
            old_tokens = words[word_id]
            frequency = word_frequencies[word_id]

            # ==================================================
            # A. 移除这个 pre-token 的旧 pair contributions
            # ==================================================

            old_unique_pairs: set[
                tuple[bytes, bytes]
            ] = set()

            for index in range(len(old_tokens) - 1):
                old_pair = (
                    old_tokens[index],
                    old_tokens[index + 1],
                )

                new_count = (
                    pair_counts[old_pair] - frequency
                )

                if new_count == 0:
                    del pair_counts[old_pair]
                else:
                    pair_counts[old_pair] = new_count

                old_unique_pairs.add(old_pair)

            # 从 pair -> word IDs 索引中移除当前 word。
            for old_pair in old_unique_pairs:
                word_ids = pair_to_words[old_pair]
                word_ids.discard(word_id)

                if not word_ids:
                    del pair_to_words[old_pair]

            # ==================================================
            # B. 在当前 pre-token 中执行 merge
            # ==================================================

            new_tokens = _merge_pair(
                old_tokens,
                best_pair,
            )

            words[word_id] = new_tokens

            # ==================================================
            # C. 加入更新后的 pair contributions
            # ==================================================

            new_unique_pairs: set[
                tuple[bytes, bytes]
            ] = set()

            for index in range(len(new_tokens) - 1):
                new_pair = (
                    new_tokens[index],
                    new_tokens[index + 1],
                )

                pair_counts[new_pair] = (
                    pair_counts.get(new_pair, 0)
                    + frequency
                )

                new_unique_pairs.add(new_pair)

            # 更新 pair -> word IDs 反向索引。
            for new_pair in new_unique_pairs:
                pair_to_words[new_pair].add(word_id)

    return vocab, merges