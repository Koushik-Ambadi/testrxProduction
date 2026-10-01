"""Pinned deterministic tokenizer used only for baseline chunk boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import re


TOKENIZER_NAME = "testrx_regex_tokenizer"
TOKENIZER_VERSION = "1.0"
TOKEN_PATTERN = re.compile(r"\w+(?:[._/-]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


class RegexTokenizer:
    """A versioned regex tokenizer with stable Unicode offsets."""

    name = TOKENIZER_NAME
    version = TOKENIZER_VERSION

    def tokenize(self, text: str) -> list[Token]:
        return [Token(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.tokenize(text))
