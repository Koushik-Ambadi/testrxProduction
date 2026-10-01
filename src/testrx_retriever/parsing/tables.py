"""Physical and logical table reconstruction for the TESTRX manual."""

from __future__ import annotations

import re
from dataclasses import asdict

from .domain import BoundingBox, PhysicalLine, PhysicalPage, SourceSpan, TableFragment
from .normalize import normalize_cell, normalize_text


TABLE_CAPTION_RE = re.compile(r"^Table\s*(\d+)\s*:\s*(.+)$", re.IGNORECASE)


def _line_overlaps_bbox(line: PhysicalLine, bbox: BoundingBox) -> bool:
    vertical = min(line.bbox.bottom, bbox.bottom) - max(line.bbox.top, bbox.top)
    horizontal = min(line.bbox.x1, bbox.x1) - max(line.bbox.x0, bbox.x0)
    return vertical > 0 and horizontal > 0


def extract_table_fragments(pdf_page, page_number: int, lines: list[PhysicalLine]) -> list[TableFragment]:
    fragments: list[TableFragment] = []
    for index, table in enumerate(pdf_page.find_tables(), start=1):
        bbox = BoundingBox.from_values(table.bbox)
        raw_rows = [
            [normalize_cell(cell) for cell in row]
            for row in table.extract()
        ]
        _recover_first_column_rowspan(pdf_page, table, raw_rows)
        column_edges = sorted(
            {
                round(float(edge), 3)
                for cell in table.cells
                for edge in (cell[0], cell[2])
            }
        )
        caption_line = _nearest_caption(lines, bbox)
        caption = caption_line.normalized_text if caption_line else None
        match = TABLE_CAPTION_RE.match(caption or "")
        fragments.append(
            TableFragment(
                id=f"p{page_number:03d}-table-{index:02d}",
                page=page_number,
                order=index,
                bbox=bbox,
                raw_rows=raw_rows,
                column_edges=column_edges,
                caption=caption,
                printed_id=match.group(1) if match else None,
            )
        )
        for line in lines:
            if _line_overlaps_bbox(line, bbox):
                line.classification = "table_text"
        if caption_line:
            caption_line.classification = "table_caption"
    return fragments


def _recover_first_column_rowspan(pdf_page, table, raw_rows: list[list[str | None]]) -> None:
    """Recover text in a visually merged first column omitted by pdfplumber."""
    if len(raw_rows) < 2 or not raw_rows[0] or len(raw_rows[0]) < 2:
        return
    if not all(not row[0] for row in raw_rows[1:] if row):
        return
    first_header_cell = table.rows[0].cells[0]
    if not first_header_cell:
        return
    left, _, right, header_bottom = first_header_cell
    if header_bottom >= table.bbox[3] - 2:
        return
    region = (left, header_bottom + 0.1, right, table.bbox[3] - 0.1)
    recovered = normalize_cell(pdf_page.within_bbox(region).extract_text() or "")
    if recovered:
        raw_rows[1][0] = recovered


def _nearest_caption(lines: list[PhysicalLine], bbox: BoundingBox) -> PhysicalLine | None:
    candidates = [
        line
        for line in lines
        if TABLE_CAPTION_RE.match(line.normalized_text)
        and -4 <= line.bbox.top - bbox.bottom <= 42
    ]
    return min(candidates, key=lambda line: abs(line.bbox.top - bbox.bottom), default=None)


def _can_continue(
    previous: TableFragment,
    current: TableFragment,
    page_height: float,
    previous_owner: str | None,
    current_owner: str | None,
) -> bool:
    return (
        current.page == previous.page + 1
        and previous.bbox.bottom >= page_height - 110
        and current.bbox.top <= 270
        and previous_owner is not None
        and previous_owner == current_owner
        and _compatible_column_geometry(previous, current)
    )


def _compatible_column_geometry(previous: TableFragment, current: TableFragment) -> bool:
    if abs(previous.bbox.x1 - current.bbox.x1) > 8:
        return False
    if not previous.column_edges or not current.column_edges:
        return False
    matched = sum(
        any(abs(current_edge - previous_edge) <= 8 for previous_edge in previous.column_edges)
        for current_edge in current.column_edges
    )
    return matched >= min(2, len(current.column_edges))


def _fragment_owner(fragment: TableFragment, pages: list[PhysicalPage]) -> str | None:
    owner = None
    for page in pages:
        if page.page_number > fragment.page:
            break
        for line in page.lines:
            if line.classification != "heading":
                continue
            if page.page_number == fragment.page and line.bbox.top > fragment.bbox.top:
                break
            match = re.match(r"^(\d+(?:\.\d+)*)\.?\s", line.normalized_text)
            if match:
                owner = match.group(1)
    return owner


def group_logical_tables(pages: list[PhysicalPage]) -> list[dict]:
    """Join fragments when a bottom-of-page table ends at the next-page caption."""
    fragments = [fragment for page in pages for fragment in page.table_fragments]
    heights = {page.page_number: page.height for page in pages}
    owners = {fragment.id: _fragment_owner(fragment, pages) for fragment in fragments}
    groups: list[list[TableFragment]] = []
    pending: list[TableFragment] = []

    for fragment in fragments:
        if fragment.caption:
            if pending and _can_continue(
                pending[-1], fragment, heights[pending[-1].page],
                owners[pending[-1].id], owners[fragment.id],
            ):
                groups.append([*pending, fragment])
                pending = []
            else:
                groups.extend([[item] for item in pending])
                pending = []
                groups.append([fragment])
        else:
            if pending and not _can_continue(
                pending[-1], fragment, heights[pending[-1].page],
                owners[pending[-1].id], owners[fragment.id],
            ):
                groups.extend([[item] for item in pending])
                pending = []
            pending.append(fragment)

    groups.extend([[item] for item in pending])
    return [_logical_table(group, index) for index, group in enumerate(groups, start=1)]


def _logical_table(fragments: list[TableFragment], sequence: int) -> dict:
    caption_fragment = next((fragment for fragment in reversed(fragments) if fragment.caption), None)
    printed_id = caption_fragment.printed_id if caption_fragment else None
    caption = caption_fragment.caption if caption_fragment else None
    first_rows = fragments[0].raw_rows
    header = first_rows[0] if first_rows else []
    header_indices = [index for index, value in enumerate(header) if value]
    columns = [header[index] for index in header_indices]

    if len(columns) < 2:
        widest = max((len(row) for fragment in fragments for row in fragment.raw_rows), default=0)
        columns = [f"column_{index + 1}" for index in range(widest)]
        header_indices = list(range(widest))

    rows: list[list[str | None]] = []
    raw_rows: list[dict] = []
    for fragment_index, fragment in enumerate(fragments):
        fragment_rows = fragment.raw_rows
        start = 1 if _looks_like_header(fragment_rows, columns) else 0
        for row_index, row in enumerate(fragment_rows[start:], start=start):
            raw_rows.append({"fragment_id": fragment.id, "row": row_index, "cells": row})
            logical = _collapse_row(row, header_indices, len(columns))
            if any(value for value in logical):
                rows.append(logical)

    resolved_rows = _forward_fill_context(rows)
    sources = [
        asdict(SourceSpan(page=fragment.page, bbox=fragment.bbox, source_ids=[fragment.id]))
        for fragment in fragments
    ]
    return {
        "id": f"table_{printed_id}" if printed_id else f"table_unidentified_{sequence:02d}",
        "printed_id": printed_id,
        "caption": caption,
        "columns": columns,
        "rows": rows,
        "resolved_rows": resolved_rows,
        "raw_rows": raw_rows,
        "page_start": fragments[0].page,
        "page_end": fragments[-1].page,
        "fragment_ids": [fragment.id for fragment in fragments],
        "sources": sources,
    }


def _looks_like_header(rows: list[list[str | None]], columns: list[str | None]) -> bool:
    if not rows:
        return False
    values = [value for value in rows[0] if value]
    return values == columns


def _collapse_row(
    row: list[str | None], header_indices: list[int], column_count: int
) -> list[str | None]:
    if len(row) == column_count:
        return list(row)
    if len(row) < column_count:
        return [None] * (column_count - len(row)) + list(row)

    values: list[str | None] = []
    for position, start in enumerate(header_indices):
        end = header_indices[position + 1] if position + 1 < len(header_indices) else len(row)
        populated = [value for value in row[start:end] if value]
        values.append(" > ".join(populated) if populated else None)
    return values


def _forward_fill_context(rows: list[list[str | None]]) -> list[list[str | None]]:
    if not rows:
        return []
    context: list[str | None] = [None] * max(len(row) for row in rows)
    resolved: list[list[str | None]] = []
    for row in rows:
        current = list(row) + [None] * (len(context) - len(row))
        for index in range(max(0, len(current) - 1)):
            if current[index]:
                context[index] = current[index]
            else:
                current[index] = context[index]
        resolved.append(current)
    return resolved


def table_caption_text(table: dict) -> str:
    return normalize_text(table.get("caption") or table["id"])
