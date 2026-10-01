"""Conservative normalization that keeps source lines untouched."""

from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"[\t\v\f ]+")


def normalize_text(text: str) -> str:
    """Normalize extraction artifacts without rewriting technical content."""
    value = unicodedata.normalize("NFC", text)
    value = value.replace("\uf0b7", "•")
    value = value.replace("\ufffe", "-").replace("\uffff", "-")
    value = value.replace("\u00ad", "")
    value = value.replace("\u2011", "-")
    value = _SPACE_RE.sub(" ", value)
    return value.strip()


def normalize_cell(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_text(value.replace("\r", " ").replace("\n", " "))
    return normalized or None


def join_text(left: str, right: str) -> str:
    """Join wrapped prose while retaining explicit word-forming hyphens."""
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and right[:1].islower():
        return left[:-1] + right
    return f"{left} {right}"


def strip_bullet(text: str) -> tuple[str, int] | None:
    value = normalize_text(text)
    if value.startswith("•"):
        return value[1:].strip(), 0
    if re.match(r"^[o○]\s+", value):
        return re.sub(r"^[o○]\s+", "", value).strip(), 1
    return None


def strip_procedure_number(text: str) -> tuple[str, str] | None:
    match = re.match(r"^(\d+)\.\s+(.+)$", normalize_text(text))
    if not match:
        return None
    return match.group(2).strip(), match.group(1)

