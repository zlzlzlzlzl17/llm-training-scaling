from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re


# 必须与 BPE training 使用相同的 GPT-2 pre-tokenization regex。
PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

PRETOKEN_PATTERN = re.compile(PAT)


# 避免每次编码都重复创建 256 个单 byte 对象。
#
# 例如：
# SINGLE_BYTE_TOKENS[97] == b"a"
SINGLE_BYTE_TOKENS: tuple[bytes, ...] = tuple(
    bytes([value])
    for value in range(256)
)


class Tokenizer:
    """
    Byte-level BPE tokenizer。

    vocab:
        token ID -> token bytes

    merges:
        按训练顺序保存的 BPE merge pairs。

    special_tokens:
        编码时必须完整保留、不能被普通 BPE 拆开的字符串。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        # 复制输入，防止修改外部传入的对象。
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)

        # None 转换为空列表，同时删除重复 special token。
        self.special_tokens: list[str] = list(
            dict.fromkeys(special_tokens or [])
        )

        # ------------------------------------------------------
        # 1. 建立反向 vocabulary：
        #
        # token bytes -> token ID
        #
        # encode() 最后需要通过 token bytes 找到 ID。
        # ------------------------------------------------------

        self.token_to_id: dict[bytes, int] = {}

        for token_id, token_bytes in self.vocab.items():
            if not isinstance(token_id, int):
                raise TypeError(
                    "Vocabulary token IDs must be integers."
                )

            if not isinstance(token_bytes, bytes):
                raise TypeError(
                    "Vocabulary token values must be bytes."
                )

            self.token_to_id[token_bytes] = token_id

        # ------------------------------------------------------
        # 2. 确保所有 special tokens 都在 vocabulary 中
        # ------------------------------------------------------

        self.special_token_to_id: dict[str, int] = {}

        next_token_id = max(
            self.vocab.keys(),
            default=-1,
        ) + 1

        for special_token in self.special_tokens:
            if special_token == "":
                raise ValueError(
                    "Special tokens cannot be empty strings."
                )

            special_bytes = special_token.encode("utf-8")

            token_id = self.token_to_id.get(special_bytes)

            # 如果 vocabulary 里还没有它，就追加一个新 ID。
            if token_id is None:
                token_id = next_token_id
                next_token_id += 1

                self.vocab[token_id] = special_bytes
                self.token_to_id[special_bytes] = token_id

            self.special_token_to_id[special_token] = token_id

        # ------------------------------------------------------
        # 3. 建立 merge priority/rank
        #
        # merge 越早产生，rank 越小，编码时优先级越高。
        #
        # merges[0] -> rank 0
        # merges[1] -> rank 1
        # ------------------------------------------------------

        self.merge_ranks: dict[
            tuple[bytes, bytes],
            int,
        ] = {}

        for rank, pair in enumerate(self.merges):
            # 正常情况下 pair 不会重复。
            # 如有重复，保留第一次出现的 rank。
            if pair not in self.merge_ranks:
                self.merge_ranks[pair] = rank

        # ------------------------------------------------------
        # 4. 创建 special-token regex
        #
        # 长 special token 必须放在短 token 前面。
        #
        # 例如同时存在：
        #   <|endoftext|>
        #   <|endoftext|><|endoftext|>
        #
        # 输入 double token 时，应优先匹配完整 double token。
        # ------------------------------------------------------

        self._special_tokens_for_matching = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        if self._special_tokens_for_matching:
            special_pattern = "|".join(
                re.escape(token)
                for token in self._special_tokens_for_matching
            )

            self._special_pattern: re.Pattern | None = re.compile(
                rf"(?:{special_pattern})"
            )
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        从训练脚本生成的 JSON-hex 文件加载 tokenizer。

        vocab.json 示例：

            {
                "0": "00",
                "97": "61",
                "256": "3c7c656e646f66746578747c3e"
            }

        merges.json 示例：

            [
                ["74", "68"],
                ["7468", "65"]
            ]

        bytes 使用十六进制保存，能无损表示任意 byte sequence。
        """

        vocab_path = Path(vocab_filepath)
        merges_path = Path(merges_filepath)

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_path}"
            )

        if not merges_path.exists():
            raise FileNotFoundError(
                f"Merges file not found: {merges_path}"
            )

        with vocab_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            serialized_vocab = json.load(file)

        vocab: dict[int, bytes] = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in serialized_vocab.items()
        }

        with merges_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            serialized_merges = json.load(file)

        merges: list[tuple[bytes, bytes]] = [
            (
                bytes.fromhex(left_hex),
                bytes.fromhex(right_hex),
            )
            for left_hex, right_hex in serialized_merges
        ]

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    def _iter_text_segments(
        self,
        text: str,
    ) -> Iterator[tuple[bool, str]]:
        """
        将文本分成：

            (False, 普通文本)
            (True, special token)

        special token 会被完整保留。

        例如：

            "A<|endoftext|><|endoftext|>B"

        如果只定义单个 special token，会得到：

            (False, "A")
            (True, "<|endoftext|>")
            (True, "<|endoftext|>")
            (False, "B")
        """

        if self._special_pattern is None:
            if text:
                yield False, text
            return

        cursor = 0

        for match in self._special_pattern.finditer(text):
            start = match.start()
            end = match.end()

            # Special token 前的普通文本。
            if cursor < start:
                yield False, text[cursor:start]

            # Special token 本身。
            yield True, match.group(0)

            cursor = end

        # 最后一个 special token 后面的普通文本。
        if cursor < len(text):
            yield False, text[cursor:]

    def _encode_pretoken_bytes(
        self,
        pretoken_bytes: bytes,
    ) -> Iterator[int]:
        """
        对一个 regex pre-token 应用 BPE merges。

        例如，初始 bytes：

            (b"t", b"h", b"e")

        merges：

            (b"t", b"h")  rank 0
            (b"th", b"e") rank 1

        最终得到：

            (b"the",)
        """

        if not pretoken_bytes:
            return

        # 初始时每个 UTF-8 byte 都是一个单独 token。
        tokens: list[bytes] = [
            SINGLE_BYTE_TOKENS[byte_value]
            for byte_value in pretoken_bytes
        ]

        while len(tokens) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            # 找出当前 token sequence 中 rank 最小的可用 pair。
            for index in range(len(tokens) - 1):
                pair = (
                    tokens[index],
                    tokens[index + 1],
                )

                rank = self.merge_ranks.get(pair)

                # 当前 pair 不在训练得到的 merges 中。
                if rank is None:
                    continue

                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            # 当前 sequence 已经没有可应用的 merge。
            if best_pair is None:
                break

            left, right = best_pair
            merged_token = left + right

            updated_tokens: list[bytes] = []
            index = 0

            # 从左到右，合并该 pair 的所有非重叠 occurrence。
            while index < len(tokens):
                can_merge = (
                    index + 1 < len(tokens)
                    and tokens[index] == left
                    and tokens[index + 1] == right
                )

                if can_merge:
                    updated_tokens.append(merged_token)
                    index += 2
                else:
                    updated_tokens.append(tokens[index])
                    index += 1

            tokens = updated_tokens

        # 把最终 token bytes 转换成 token IDs。
        for token_bytes in tokens:
            token_id = self.token_to_id.get(token_bytes)

            if token_id is None:
                raise ValueError(
                    "BPE produced a token that is missing from "
                    f"the vocabulary: {token_bytes!r}"
                )

            yield token_id

    def _encode_normal_text(
        self,
        text: str,
    ) -> Iterator[int]:
        """
        编码不包含 special token 的普通文本。

        先执行 GPT-2 regex pre-tokenization，
        再对每个 pre-token 独立应用 BPE。
        """

        for match in PRETOKEN_PATTERN.finditer(text):
            pretoken_text = match.group(0)
            pretoken_bytes = pretoken_text.encode("utf-8")

            yield from self._encode_pretoken_bytes(
                pretoken_bytes
            )

    def _encode_text_iter(
        self,
        text: str,
    ) -> Iterator[int]:
        """
        编码一段文本，并逐个产生 token ID。

        处理顺序：

            1. 先识别和分离 special tokens；
            2. special token 直接产生一个 ID；
            3. 普通文本使用 regex pre-tokenization；
            4. 每个普通 pre-token 应用 BPE。
        """

        for is_special, segment in self._iter_text_segments(text):
            if not segment:
                continue

            if is_special:
                token_id = self.special_token_to_id.get(segment)

                if token_id is None:
                    raise ValueError(
                        f"Unregistered special token: {segment!r}"
                    )

                yield token_id
            else:
                yield from self._encode_normal_text(segment)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        将字符串编码成完整的 token-ID list。
        """

        if not isinstance(text, str):
            raise TypeError(
                "Tokenizer.encode expects a string, "
                f"received {type(text).__name__}."
            )

        return list(self._encode_text_iter(text))

    def _iter_unit_spans(
        self,
        text: str,
    ) -> Iterator[tuple[int, int]]:
        """
        返回普通 pre-token 和 special token 在原字符串中的位置。

        encode_iterable() 用它找到一个安全的流式切分位置。
        """

        if self._special_pattern is None:
            for match in PRETOKEN_PATTERN.finditer(text):
                yield match.start(), match.end()
            return

        cursor = 0

        for special_match in self._special_pattern.finditer(text):
            special_start = special_match.start()
            special_end = special_match.end()

            # 扫描 special token 前的普通文本。
            if cursor < special_start:
                for match in PRETOKEN_PATTERN.finditer(
                    text,
                    cursor,
                    special_start,
                ):
                    yield match.start(), match.end()

            yield special_start, special_end
            cursor = special_end

        # 扫描最后一个 special token 后的普通文本。
        if cursor < len(text):
            for match in PRETOKEN_PATTERN.finditer(
                text,
                cursor,
                len(text),
            ):
                yield match.start(), match.end()

    def _partial_special_token_start(
        self,
        text: str,
    ) -> int:
        """
        检查文本尾部是否可能是某个 special token 的未完成前缀。

        返回必须保留的起始位置；没有 partial special token 时，
        返回 len(text)。

        例如 special token 是：

            <|endoftext|>

        当前 buffer 结尾是：

            abc<|endo

        则必须保留：

            <|endo
        """

        earliest_start = len(text)

        for special_token in self._special_tokens_for_matching:
            # 只检查 proper prefix；完整 special token 会被正常 regex 匹配。
            maximum_prefix_length = min(
                len(text),
                len(special_token) - 1,
            )

            for prefix_length in range(
                maximum_prefix_length,
                0,
                -1,
            ):
                prefix = special_token[:prefix_length]

                if text.endswith(prefix):
                    earliest_start = min(
                        earliest_start,
                        len(text) - prefix_length,
                    )
                    break

        return earliest_start

    def _safe_stream_prefix_length(
        self,
        buffer: str,
    ) -> int:
        """
        找出 buffer 中可以安全编码的前缀长度。

        最后一个 regex pre-token 可能继续延伸到下一个 chunk，
        所以暂时保留最后一个 token unit。

        同时保留可能跨 chunk 的 partial special token。
        """

        if not buffer:
            return 0

        last_unit_start: int | None = None

        for start, _ in self._iter_unit_spans(buffer):
            last_unit_start = start

        if last_unit_start is None:
            return 0

        partial_special_start = (
            self._partial_special_token_start(buffer)
        )

        return min(
            last_unit_start,
            partial_special_start,
        )

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        """
        流式编码字符串 iterable。

        不会一次性读取完整文件；只保留可能跨 chunk 的最后一个
        pre-token 或未完成 special token。
        """

        buffer = ""

        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError(
                    "Tokenizer.encode_iterable expects strings, "
                    f"received {type(chunk).__name__}."
                )

            if not chunk:
                continue

            buffer += chunk

            safe_length = self._safe_stream_prefix_length(
                buffer
            )

            if safe_length <= 0:
                continue

            safe_text = buffer[:safe_length]

            yield from self._encode_text_iter(safe_text)

            buffer = buffer[safe_length:]

        # iterable 已结束，剩余内容不可能再继续延伸。
        if buffer:
            yield from self._encode_text_iter(buffer)

    def decode(
        self,
        ids: list[int],
    ) -> str:
        """
        将 token IDs 解码成字符串。

        所有 token bytes 先拼接，再统一进行 UTF-8 decode。

        如果 byte sequence 不是合法 UTF-8，则使用 Unicode
        replacement character U+FFFD。
        """

        token_parts: list[bytes] = []

        for token_id in ids:
            token_bytes = self.vocab.get(token_id)

            if token_bytes is None:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            token_parts.append(token_bytes)

        combined_bytes = b"".join(token_parts)

        return combined_bytes.decode(
            "utf-8",
            errors="replace",
        )