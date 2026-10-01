"""Deterministic token-window baseline chunker with complete source lineage."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable

from ..configuration import ChunkingConfig
from .tokenization import RegexTokenizer


@dataclass(frozen=True)
class SourceSegment:
    text: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    semantic_unit_ids: tuple[str, ...]
    source_element_ids: tuple[str, ...]
    content_types: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    token_start: int
    token_end: int
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    section_paths: tuple[tuple[str, ...], ...]
    semantic_unit_ids: tuple[str, ...]
    source_element_ids: tuple[str, ...]
    content_types: tuple[str, ...]
    source_file: str = ""
    chunking_strategy: str = "token_window"
    chunking_config: tuple[tuple[str, Any], ...] = ()
    hierarchy_path: tuple[str, ...] = ()
    hierarchy_level: str | None = None
    parent_node_id: str | None = None
    split_reason: str | None = None
    fallback_split: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.chunking_strategy == "token_window" and self.hierarchy_level is None:
            for key in (
                "source_file", "chunking_strategy", "chunking_config", "hierarchy_path",
                "hierarchy_level", "parent_node_id", "split_reason", "fallback_split",
            ):
                value.pop(key)
        return value


def _stable_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _table_text(element: dict[str, Any]) -> str:
    data = element.get("data") or {}
    lines = [element.get("text") or data.get("caption") or "Table"]
    columns = data.get("columns") or []
    if columns:
        lines.append(" | ".join(str(value or "") for value in columns))
    for row in data.get("resolved_rows") or data.get("rows") or []:
        lines.append(" | ".join(str(value or "") for value in row))
    return "\n".join(lines)


def _element_text(element: dict[str, Any]) -> str:
    if element["type"] == "table":
        return _table_text(element)
    if element["type"] == "figure":
        printed_id = (element.get("data") or {}).get("printed_id")
        prefix = f"Figure Snp.{printed_id}: " if printed_id else "Figure: "
        return prefix + (element.get("text") or "")
    return element.get("text") or ""


def flatten_document(document: dict[str, Any]) -> list[SourceSegment]:
    """Flatten canonical sections and recursive elements in source order."""
    segments: list[SourceSegment] = []
    for section in sorted(document["sections"], key=lambda item: item["sequence"]):
        path = tuple(section["ancestor_path"])
        segments.append(SourceSegment(path[-1], None, None, path, (), (), ("heading",)))

        def visit(element: dict[str, Any], ancestors: tuple[str, ...] = ()) -> None:
            semantic_ids = ancestors + (element["id"],)
            text = _element_text(element).strip()
            if text:
                segments.append(
                    SourceSegment(
                        text=text,
                        page_start=int(element["page_start"]),
                        page_end=int(element["page_end"]),
                        section_path=path,
                        semantic_unit_ids=semantic_ids,
                        source_element_ids=(element["id"],),
                        content_types=(element["type"],),
                    )
                )
            for child in sorted(element.get("children", []), key=lambda item: item["sequence"]):
                visit(child, semantic_ids)

        for element in sorted(section["elements"], key=lambda item: item["sequence"]):
            visit(element)
    return segments


class TokenChunker:
    def __init__(self, config: ChunkingConfig, tokenizer: RegexTokenizer | None = None):
        config.validate()
        self.config = config
        self.tokenizer = tokenizer or RegexTokenizer()

    def chunk_document(self, document: dict[str, Any]) -> list[Chunk]:
        segments = flatten_document(document)
        if not segments:
            return []
        separator = "\n\n"
        starts: list[int] = []
        cursor = 0
        parts: list[str] = []
        for segment in segments:
            starts.append(cursor)
            parts.append(segment.text)
            cursor += len(segment.text) + len(separator)
        flat_text = separator.join(parts)
        tokens = self.tokenizer.tokenize(flat_text)
        if not tokens:
            return []

        def segment_for_offset(offset: int) -> SourceSegment:
            return segments[max(0, bisect_right(starts, offset) - 1)]

        step = self.config.chunk_size - self.config.chunk_overlap
        chunks: list[Chunk] = []
        for chunk_index, token_start in enumerate(range(0, len(tokens), step)):
            token_end = min(token_start + self.config.chunk_size, len(tokens))
            window = tokens[token_start:token_end]
            contributing = _stable_unique(segment_for_offset(token.start) for token in window)
            evidenced = tuple(segment for segment in contributing if segment.page_start is not None)
            if not evidenced:
                continue
            text = flat_text[window[0].start : window[-1].end].strip()
            section_paths = _stable_unique(segment.section_path for segment in evidenced)
            semantic_ids = _stable_unique(
                semantic_id for segment in evidenced for semantic_id in segment.semantic_unit_ids
            )
            element_ids = _stable_unique(
                element_id for segment in evidenced for element_id in segment.source_element_ids
            )
            content_types = _stable_unique(
                content_type for segment in evidenced for content_type in segment.content_types
            )
            pages_start = [segment.page_start for segment in evidenced if segment.page_start is not None]
            pages_end = [segment.page_end for segment in evidenced if segment.page_end is not None]
            identity = json.dumps(
                {
                    "document_sha256": document["source_sha256"],
                    "tokenizer": [self.tokenizer.name, self.tokenizer.version],
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "chunk_index": chunk_index,
                    "text": text,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(identity).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=f"CH-{chunk_index:05d}-{digest}",
                    document_id=document["document_id"],
                    chunk_index=chunk_index,
                    text=text,
                    token_count=len(window),
                    token_start=token_start,
                    token_end=token_end,
                    page_start=min(pages_start),
                    page_end=max(pages_end),
                    section_path=section_paths[0],
                    section_paths=section_paths,
                    semantic_unit_ids=semantic_ids,
                    source_element_ids=element_ids,
                    content_types=content_types,
                )
            )
            if token_end == len(tokens):
                break
        return chunks
