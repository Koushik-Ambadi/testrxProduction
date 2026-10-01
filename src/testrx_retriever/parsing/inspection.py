"""Human inspection output and concise parsing ambiguity records."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .domain import CanonicalDocument, Element, Section
from .structure import IMPERATIVE_VERBS


def walk_elements(elements: Iterable[Element]) -> Iterable[Element]:
    for element in elements:
        yield element
        yield from walk_elements(element.children)


def collect_parsing_warnings(document: CanonicalDocument) -> dict:
    items: list[dict] = []

    def add(kind: str, message: str, *, section_id: str | None = None,
            element: Element | None = None, pages: list[int] | None = None,
            severity: str = "warning") -> None:
        items.append(
            {
                "id": f"warning-{len(items) + 1:03d}",
                "type": kind,
                "severity": severity,
                "message": message,
                "section_id": section_id,
                "element_id": element.id if element else None,
                "pages": pages or (
                    list(range(element.page_start, element.page_end + 1)) if element else []
                ),
            }
        )

    if document.metadata.get("source_type") != "native_pdf":
        add(
            "unsupported_source_type",
            "Native text coverage is insufficient; OCR is not implemented.",
            pages=list(range(1, len(document.pages) + 1)),
            severity="error",
        )

    for section in document.sections:
        for element in walk_elements(section.elements):
            if element.type == "local_group" and not element.children:
                add("local_group_without_content", "Local group has no owned content.", section_id=section.section_id, element=element)
            if element.type == "procedure":
                values = element.data.get("items", [])
                if len(values) == 1:
                    text = str(values[0].get("text", "")).strip()
                    first = text.split()[0].lower().strip(".,:;()") if text else ""
                    if len(text.split()) <= 7 and first not in IMPERATIVE_VERBS:
                        add("ambiguous_numbered_item", "Short numbered item remains classified as a procedure.", section_id=section.section_id, element=element)
            if element.type == "paragraph" and element.page_end > element.page_start:
                add("possible_cross_page_paragraph_merge", "Paragraph was joined across a page boundary.", section_id=section.section_id, element=element)
            if element.type == "table":
                fragments = element.data.get("fragment_ids", [])
                if not element.data.get("printed_id"):
                    add("unidentified_table", "Detected table has no printed table identifier.", section_id=section.section_id, element=element)
                if len(fragments) > 1:
                    add("possible_table_continuation", "Multiple physical fragments were joined into one logical table; merge passed structural checks.", section_id=section.section_id, element=element)
            if element.type == "figure" and not element.data.get("region_id"):
                add("unassociated_figure", "Caption has no qualifying raster image region; source may use vector/compound artwork.", section_id=section.section_id, element=element)

    counts = Counter(item["type"] for item in items)
    severity_counts = Counter(item["severity"] for item in items)
    return {
        "summary": {
            "total": len(items),
            "warnings": severity_counts["warning"],
            "errors": severity_counts["error"],
            "by_type": dict(sorted(counts.items())),
        },
        "items": items,
    }


def render_semantic_structure(document: CanonicalDocument) -> str:
    lines = [
        f"# {document.title} - Parsed Semantic Structure",
        "",
        f"Source: `{document.source_file}`",
        f"Pages: {len(document.pages)}",
        "",
    ]
    for section in document.sections:
        indent = "    " * (section.level - 1)
        lines.append(f"{indent}L{section.level}  {section.section_id} {section.title}")
        if section.page_end > section.page_start:
            lines.append(f"{indent}    [pages {section.page_start} -> {section.page_end}]")
        lines.append("")
        for element in section.elements:
            _render_element(lines, element, section.level, 1)
    return "\n".join(lines).rstrip() + "\n"


def _render_element(lines: list[str], element: Element, section_level: int, group_depth: int) -> None:
    indent = "    " * section_level
    if element.type in {"local_group", "labelled_block"}:
        prefix = f"G{group_depth}"
        lines.append(f"{indent}{prefix}  {element.text}")
        if element.page_end > element.page_start:
            lines.append(f"{indent}    [pages {element.page_start} -> {element.page_end}]")
        lines.append("")
        for child in element.children:
            _render_element(lines, child, section_level + 1, group_depth + 1)
        return

    if element.type == "paragraph":
        lines.extend([f"{indent}[PARAGRAPH]", f"{indent}{element.text}", ""])
        return
    if element.type == "list":
        lines.append(f"{indent}[LIST]")
        for item in element.data.get("items", []):
            item_indent = indent + "    " * int(item.get("level", 0))
            lines.append(f"{item_indent}- {item.get('text', '')}")
        lines.append("")
        return
    if element.type == "procedure":
        lines.append(f"{indent}[PROCEDURE]")
        for item in element.data.get("items", []):
            lines.append(f"{indent}{item.get('marker', '')}. {item.get('text', '')}")
        lines.append("")
        return
    if element.type == "figure":
        lines.extend(
            [f"{indent}[FIGURE]", f"{indent}Snp.{element.data.get('printed_id')} - {element.text}", ""]
        )
        return
    if element.type == "table":
        caption = str(element.text or element.id).replace(":", " -", 1)
        lines.append(f"{indent}[TABLE] {caption}")
        if element.page_end > element.page_start:
            lines.append(f"{indent}[pages {element.page_start} -> {element.page_end}]")
        fragments = element.data.get("fragment_ids", [])
        if len(fragments) > 1:
            lines.append(f"{indent}[{len(fragments)} physical fragments -> 1 logical table]")
        for row in element.data.get("resolved_rows", []):
            values = [str(value) for value in row if value is not None]
            if not values:
                continue
            relation = " > ".join(values[:-1]) if len(values) > 1 else values[0]
            lines.append(f"{indent}{relation}")
            if len(values) > 1:
                lines.append(f"{indent}    {values[-1]}")
        lines.append("")
