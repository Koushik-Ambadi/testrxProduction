"""Small canonical data model used by both extraction and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x0: float
    top: float
    x1: float
    bottom: float

    @classmethod
    def from_values(cls, values: tuple[float, float, float, float]) -> "BoundingBox":
        return cls(*(round(float(value), 3) for value in values))


@dataclass
class SourceSpan:
    page: int
    bbox: BoundingBox
    source_ids: list[str]


@dataclass
class PhysicalLine:
    id: str
    page: int
    order: int
    text: str
    normalized_text: str
    bbox: BoundingBox
    font_name: str
    font_size: float
    classification: str = "body"


@dataclass
class TableFragment:
    id: str
    page: int
    order: int
    bbox: BoundingBox
    raw_rows: list[list[str | None]]
    column_edges: list[float] = field(default_factory=list)
    caption: str | None = None
    printed_id: str | None = None


@dataclass
class FigureRegion:
    id: str
    page: int
    bbox: BoundingBox


@dataclass
class PhysicalPage:
    page_number: int
    width: float
    height: float
    page_type: str
    lines: list[PhysicalLine] = field(default_factory=list)
    table_fragments: list[TableFragment] = field(default_factory=list)
    figure_regions: list[FigureRegion] = field(default_factory=list)


@dataclass
class Element:
    id: str
    type: str
    sequence: int
    text: str | None
    page_start: int
    page_end: int
    sources: list[SourceSpan]
    data: dict[str, Any] = field(default_factory=dict)
    children: list["Element"] = field(default_factory=list)


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    parent_section_id: str | None
    ancestor_path: list[str]
    sequence: int
    page_start: int
    page_end: int
    heading_source: SourceSpan
    elements: list[Element] = field(default_factory=list)


@dataclass
class CanonicalDocument:
    schema_version: str
    parser_version: str
    document_id: str
    title: str
    source_file: str
    source_sha256: str
    metadata: dict[str, Any]
    pages: list[PhysicalPage]
    sections: list[Section]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
