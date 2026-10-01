"""Hierarchy-aware chunking and raw canonical hierarchy diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from statistics import mean, median
from typing import Any, Iterable

from ..configuration import ChunkingConfig
from .chunking import Chunk, SourceSegment, _element_text, _stable_unique
from .tokenization import RegexTokenizer, Token


@dataclass
class HierarchyNode:
    """One canonical section or semantic element and its descendants."""

    node_id: str
    node_type: str
    hierarchy_level: str
    label: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    parent_node_id: str | None
    parent_labels: tuple[str, ...]
    semantic_ancestors: tuple[str, ...] = ()
    own_segment: SourceSegment | None = None
    children: list["HierarchyNode"] = field(default_factory=list)
    order_key: tuple[int, int, int] = (0, 0, 0)

    def subtree_segments(self) -> tuple[SourceSegment, ...]:
        segments: list[SourceSegment] = []
        if self.own_segment is not None:
            segments.append(self.own_segment)
        for child in sorted(self.children, key=lambda item: item.order_key):
            segments.extend(child.subtree_segments())
        return tuple(segments)


def _label(text: str, fallback: str) -> str:
    compact = " ".join(text.split())
    if not compact:
        return fallback
    return compact if len(compact) <= 160 else compact[:157].rstrip() + "..."


def build_hierarchy(document: dict[str, Any]) -> HierarchyNode:
    """Build a deterministic tree from canonical section and element ownership."""
    root = HierarchyNode(
        node_id=str(document["document_id"]),
        node_type="document",
        hierarchy_level="document",
        label=str(document.get("title") or document["document_id"]),
        page_start=None,
        page_end=None,
        section_path=(),
        parent_node_id=None,
        parent_labels=(),
    )
    sections: dict[str, HierarchyNode] = {}

    def element_node(
        element: dict[str, Any], section_path: tuple[str, ...], parent_id: str,
        parent_labels: tuple[str, ...], semantic_ancestors: tuple[str, ...],
    ) -> HierarchyNode:
        element_id = str(element["id"])
        element_type = str(element["type"])
        text = _element_text(element).strip()
        semantic_ids = semantic_ancestors + (element_id,)
        segment = SourceSegment(
            text=text,
            page_start=int(element["page_start"]),
            page_end=int(element["page_end"]),
            section_path=section_path,
            semantic_unit_ids=semantic_ids,
            source_element_ids=(element_id,),
            content_types=(element_type,),
        ) if text else None
        node_label = _label(text, element_type)
        node = HierarchyNode(
            node_id=element_id,
            node_type=element_type,
            hierarchy_level=f"element:{element_type}",
            label=node_label,
            page_start=int(element["page_start"]),
            page_end=int(element["page_end"]),
            section_path=section_path,
            parent_node_id=parent_id,
            parent_labels=parent_labels,
            semantic_ancestors=semantic_ancestors,
            own_segment=segment,
            order_key=(int(element["page_start"]), 1, int(element["sequence"])),
        )
        node.children = [
            element_node(child, section_path, element_id, parent_labels + (node_label,), semantic_ids)
            for child in sorted(element.get("children", []), key=lambda item: item["sequence"])
        ]
        return node

    for section in sorted(document["sections"], key=lambda item: item["sequence"]):
        path = tuple(section["ancestor_path"])
        section_id = str(section["section_id"])
        title = str(section.get("title") or path[-1])
        heading = SourceSegment(
            text=path[-1], page_start=None, page_end=None, section_path=path,
            semantic_unit_ids=(), source_element_ids=(), content_types=("heading",),
        )
        node = HierarchyNode(
            node_id=section_id,
            node_type="section",
            hierarchy_level=f"section_level_{int(section['level'])}",
            label=title,
            page_start=int(section["page_start"]),
            page_end=int(section["page_end"]),
            section_path=path,
            parent_node_id=section.get("parent_section_id") or str(document["document_id"]),
            parent_labels=path[:-1],
            own_segment=heading,
            order_key=(int(section["page_start"]), 0, int(section["sequence"])),
        )
        node.children = [
            element_node(element, path, section_id, path, ())
            for element in sorted(section.get("elements", []), key=lambda item: item["sequence"])
        ]
        sections[section_id] = node

    for section in sorted(document["sections"], key=lambda item: item["sequence"]):
        node = sections[str(section["section_id"])]
        parent_id = section.get("parent_section_id")
        if parent_id:
            sections[str(parent_id)].children.append(node)
        else:
            root.children.append(node)
    return root


def iter_hierarchy(root: HierarchyNode, include_document: bool = True) -> Iterable[HierarchyNode]:
    if include_document:
        yield root
    for child in sorted(root.children, key=lambda item: item.order_key):
        yield child
        yield from iter_hierarchy(child, include_document=False)


def _segments_text(segments: Iterable[SourceSegment]) -> str:
    return "\n\n".join(segment.text for segment in segments if segment.text.strip()).strip()


def hierarchy_node_records(
    document: dict[str, Any], tokenizer: RegexTokenizer | None = None,
) -> list[dict[str, Any]]:
    """Return raw, deterministic per-node subtree measurements."""
    tokenizer = tokenizer or RegexTokenizer()
    root = build_hierarchy(document)
    records: list[dict[str, Any]] = []
    for node in iter_hierarchy(root):
        text = _segments_text(node.subtree_segments())
        records.append({
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "node_type": node.node_type,
            "hierarchy_level": node.hierarchy_level,
            "hierarchy_path": list(node.parent_labels + (node.label,)),
            "section_path": list(node.section_path),
            "page_start": node.page_start,
            "page_end": node.page_end,
            "child_count": len(node.children),
            "token_count": tokenizer.count(text),
        })
    return records


def _percentile(values: list[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def token_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize raw token counts by hierarchy level using nearest-rank percentiles."""
    thresholds = (256, 384, 512, 768, 1024, 2048, 4096)
    grouped: dict[str, list[int]] = {}
    for record in records:
        grouped.setdefault(str(record["hierarchy_level"]), []).append(int(record["token_count"]))
    levels: list[dict[str, Any]] = []
    for level, values in grouped.items():
        levels.append({
            "level": level,
            "node_count": len(values),
            "min_tokens": min(values),
            "max_tokens": max(values),
            "avg_tokens": round(mean(values), 6),
            "median_tokens": median(values),
            "p90_tokens": _percentile(values, 0.90),
            "p95_tokens": _percentile(values, 0.95),
            "p99_tokens": _percentile(values, 0.99),
            "nodes_over_limit": {str(limit): sum(value > limit for value in values) for limit in thresholds},
        })
    return {"percentile_method": "nearest_rank", "levels": levels}


def chunk_statistics(chunks: list[Chunk]) -> dict[str, Any]:
    """Return deterministic size and provenance counts for one emitted chunk set."""
    values = [chunk.token_count for chunk in chunks]
    by_level: dict[str, int] = {}
    for chunk in chunks:
        level = chunk.hierarchy_level or "token_window"
        by_level[level] = by_level.get(level, 0) + 1
    return {
        "chunk_count": len(chunks),
        "min_tokens": min(values) if values else 0,
        "max_tokens": max(values) if values else 0,
        "avg_tokens": round(mean(values), 6) if values else 0.0,
        "median_tokens": median(values) if values else 0,
        "p90_tokens": _percentile(values, 0.90),
        "p95_tokens": _percentile(values, 0.95),
        "chunks_at_each_hierarchy_level": by_level,
        "fallback_split_count": sum(chunk.fallback_split for chunk in chunks),
    }


def render_hierarchy_summary(statistics: dict[str, Any]) -> str:
    lines = ["# TESTRX pure hierarchy token statistics", ""]
    for item in statistics["levels"]:
        lines.extend([
            f"## {item['level']}", "",
            f"- Nodes: {item['node_count']}",
            f"- Tokens min / average / median / max: {item['min_tokens']} / {item['avg_tokens']:.2f} / {item['median_tokens']} / {item['max_tokens']}",
            f"- Tokens p90 / p95 / p99: {item['p90_tokens']} / {item['p95_tokens']} / {item['p99_tokens']}",
            "- Nodes over limits: " + ", ".join(
                f">{limit}: {count}" for limit, count in item["nodes_over_limit"].items()
            ), "",
        ])
    return "\n".join(lines)


def _context_text(node: HierarchyNode) -> str:
    labels = node.parent_labels
    if not labels:
        return ""
    return "Hierarchy: " + " > ".join(labels)


def _lineage(node: HierarchyNode) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    segments = node.subtree_segments()
    semantic = _stable_unique(
        item for segment in segments for item in segment.semantic_unit_ids
    )
    semantic = _stable_unique(node.semantic_ancestors + semantic)
    elements = _stable_unique(
        item for segment in segments for item in segment.source_element_ids
    )
    content_types = _stable_unique(
        item for segment in segments for item in segment.content_types if item != "heading"
    )
    return semantic, elements, content_types


class HierarchicalChunker:
    """Emit every hierarchy subtree or descend until it fits a token maximum."""

    def __init__(self, config: ChunkingConfig, tokenizer: RegexTokenizer | None = None):
        config.validate()
        if config.strategy not in {"hierarchical_pure", "hierarchical_max_tokens"}:
            raise ValueError("HierarchicalChunker requires a hierarchical strategy")
        self.config = config
        self.tokenizer = tokenizer or RegexTokenizer()

    def _make_chunk(
        self, document: dict[str, Any], node: HierarchyNode, text: str,
        chunk_index: int, split_reason: str | None = None,
        fallback_split: bool = False, part_index: int | None = None,
    ) -> Chunk:
        tokens = self.tokenizer.tokenize(text)
        semantic, elements, content_types = _lineage(node)
        identity = json.dumps({
            "document_sha256": document["source_sha256"],
            "tokenizer": [self.tokenizer.name, self.tokenizer.version],
            "strategy": self.config.strategy,
            "max_tokens": self.config.max_tokens,
            "node_id": node.node_id,
            "part_index": part_index,
            "text": text,
        }, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:16]
        return Chunk(
            chunk_id=f"CH-{chunk_index:05d}-{digest}",
            document_id=str(document["document_id"]),
            chunk_index=chunk_index,
            text=text,
            token_count=len(tokens),
            token_start=0,
            token_end=len(tokens),
            page_start=int(node.page_start or 1),
            page_end=int(node.page_end or node.page_start or 1),
            section_path=node.section_path,
            section_paths=(node.section_path,),
            semantic_unit_ids=semantic,
            source_element_ids=elements,
            content_types=content_types,
            source_file=str(document.get("source_file", "")),
            chunking_strategy=self.config.strategy,
            chunking_config=(("max_tokens", self.config.max_tokens),),
            hierarchy_path=node.parent_labels + (node.label,),
            hierarchy_level=node.hierarchy_level,
            parent_node_id=node.parent_node_id,
            split_reason=split_reason,
            fallback_split=fallback_split,
        )

    def _node_text(self, node: HierarchyNode) -> str:
        content = _segments_text(node.subtree_segments())
        context = _context_text(node)
        return "\n\n".join(part for part in (context, content) if part).strip()

    def _fallback_chunks(
        self, document: dict[str, Any], node: HierarchyNode, start_index: int,
    ) -> list[Chunk]:
        max_tokens = int(self.config.max_tokens or 0)
        content = _segments_text(node.subtree_segments())
        content_tokens = self.tokenizer.tokenize(content)
        context = _context_text(node)
        context_tokens = self.tokenizer.tokenize(context)
        if len(context_tokens) >= max_tokens:
            context = ""
            context_tokens = []
        available = max_tokens - len(context_tokens)
        chunks: list[Chunk] = []
        for part_index, start in enumerate(range(0, len(content_tokens), available)):
            window = content_tokens[start:start + available]
            fragment = content[window[0].start:window[-1].end].strip()
            text = "\n\n".join(part for part in (context, fragment) if part)
            chunks.append(self._make_chunk(
                document, node, text, start_index + part_index,
                split_reason="max_tokens_oversized_leaf", fallback_split=True,
                part_index=part_index,
            ))
        return chunks

    def chunk_document(self, document: dict[str, Any]) -> list[Chunk]:
        root = build_hierarchy(document)
        if self.config.strategy == "hierarchical_pure":
            nodes = list(iter_hierarchy(root, include_document=False))
            return [
                self._make_chunk(document, node, self._node_text(node), index)
                for index, node in enumerate(nodes) if self._node_text(node)
            ]

        chunks: list[Chunk] = []

        def visit(node: HierarchyNode, parent_exceeded_max: bool = False) -> None:
            text = self._node_text(node)
            if not text:
                return
            if self.tokenizer.count(text) <= int(self.config.max_tokens or 0):
                chunks.append(self._make_chunk(
                    document, node, text, len(chunks),
                    split_reason="parent_exceeded_max" if parent_exceeded_max else None,
                ))
                return
            if node.children:
                for child in sorted(node.children, key=lambda item: item.order_key):
                    visit(child, parent_exceeded_max=True)
                return
            chunks.extend(self._fallback_chunks(document, node, len(chunks)))

        for child in sorted(root.children, key=lambda item: item.order_key):
            visit(child)
        return chunks


def build_chunker(config: ChunkingConfig, tokenizer: RegexTokenizer | None = None) -> Any:
    """Construct a configured chunker without changing legacy imports."""
    if config.strategy == "token_window":
        from .chunking import TokenChunker
        return TokenChunker(config, tokenizer)
    return HierarchicalChunker(config, tokenizer)
