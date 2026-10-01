"""Deterministic reconstruction of sections and semantic elements."""

from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from .figures import figure_from_caption
from .domain import BoundingBox, Element, PhysicalLine, PhysicalPage, Section, SourceSpan
from .normalize import join_text, normalize_text, strip_bullet, strip_procedure_number


HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+(.+)$")
IMPERATIVE_VERBS = {
    "add", "choose", "click", "configure", "create", "delete", "double-click",
    "enter", "export", "go", "import", "load", "navigate", "open", "press",
    "remove", "return", "right-click", "save", "select", "set", "verify",
}


def _source(line: PhysicalLine) -> SourceSpan:
    return SourceSpan(page=line.page, bbox=line.bbox, source_ids=[line.id])


def classify_headings(pages: list[PhysicalPage]) -> list[tuple[PhysicalLine, str, str]]:
    headings: list[tuple[PhysicalLine, str, str]] = []
    for page in pages:
        if page.page_type != "body":
            continue
        for line in page.lines:
            if line.classification not in {"body", "heading"}:
                continue
            match = HEADING_RE.match(line.normalized_text)
            is_light_font = "light" in line.font_name.lower()
            depth = len(match.group(1).split(".")) if match else 0
            uses_heading_geometry = bool(
                match
                and (
                    is_light_font
                    or (depth == 1 and line.font_size >= 15)
                    or (depth == 2 and line.font_size >= 12.5)
                    or (depth >= 3 and line.bbox.x0 <= 120)
                )
            )
            if line.classification == "heading" or (match and uses_heading_geometry):
                assert match is not None
                section_id = match.group(1)
                title = match.group(2).strip()
                line.classification = "heading"
                headings.append((line, section_id, title))
    return headings


def build_sections(
    pages: list[PhysicalPage], logical_tables: list[dict[str, Any]]
) -> list[Section]:
    heading_records = classify_headings(pages)
    by_id: dict[str, Section] = {}
    positions: list[tuple[tuple[int, float], Section]] = []
    sections: list[Section] = []

    for sequence, (line, section_id, title) in enumerate(heading_records, start=1):
        parts = section_id.split(".")
        parent_id = ".".join(parts[:-1]) or None
        if parent_id not in by_id:
            parent_id = _nearest_existing_parent(parts, by_id)
        parent_path = by_id[parent_id].ancestor_path if parent_id else []
        display = f"{section_id} {title}"
        section = Section(
            section_id=section_id,
            title=title,
            level=len(parts),
            parent_section_id=parent_id,
            ancestor_path=[*parent_path, display],
            sequence=sequence,
            page_start=line.page,
            page_end=line.page,
            heading_source=_source(line),
        )
        sections.append(section)
        by_id[section_id] = section
        positions.append(((line.page, line.bbox.top), section))

    events: list[tuple[tuple[int, float, int], str, Any, Section]] = []
    for page in pages:
        if page.page_type != "body":
            continue
        for line in page.lines:
            if line.classification != "body":
                continue
            owner = _owner_for_position(positions, line.page, line.bbox.top)
            if owner:
                events.append(((line.page, line.bbox.top, 1), "line", line, owner))
        for line in page.lines:
            if line.classification != "figure_caption":
                continue
            owner = _owner_for_position(positions, line.page, line.bbox.top)
            if owner:
                figure = figure_from_caption(line, page.figure_regions)
                events.append(((line.page, line.bbox.top, 2), "figure", figure, owner))

    for table in logical_tables:
        first_source = table["sources"][0]
        owner = _owner_for_position(
            positions, int(first_source["page"]), float(first_source["bbox"]["top"])
        )
        if owner:
            events.append(
                (
                    (int(first_source["page"]), float(first_source["bbox"]["top"]), 0),
                    "table",
                    table,
                    owner,
                )
            )

    events.sort(key=lambda item: item[0])
    element_sequence = 0
    current: dict[str, Any] | None = None
    current_owner: Section | None = None

    def flush() -> None:
        nonlocal current, current_owner, element_sequence
        if not current or not current_owner:
            current = None
            current_owner = None
            return
        element_sequence += 1
        current_owner.elements.append(_materialize(current, element_sequence, current_owner))
        current = None
        current_owner = None

    for _, kind, payload, owner in events:
        if kind != "line":
            flush()
            element_sequence += 1
            if kind == "table":
                owner.elements.append(_table_element(payload, element_sequence, owner))
            else:
                owner.elements.append(_figure_element(payload, element_sequence, owner, pages))
            continue

        line: PhysicalLine = payload
        bullet = strip_bullet(line.normalized_text)
        procedure = None if "light" in line.font_name.lower() else strip_procedure_number(line.normalized_text)

        if bullet or procedure:
            ordered = procedure is not None
            item_text = procedure[0] if procedure else bullet[0]
            level = 0 if procedure else bullet[1]
            marker = procedure[1] if procedure else "bullet"
            desired = "procedure" if ordered else "list"
            if not current or current_owner != owner or current["kind"] != desired:
                flush()
                current_owner = owner
                current = {"kind": desired, "items": [], "last_line": line, "base_x": line.bbox.x0}
            current["items"].append(
                {"text": item_text, "level": level, "marker": marker, "sources": [_source(line)]}
            )
            current["last_line"] = line
            continue

        if current and current_owner == owner and current["kind"] in {"list", "procedure"}:
            last_line: PhysicalLine = current["last_line"]
            gap = line.bbox.top - last_line.bbox.bottom if line.page == last_line.page else 0
            is_continuation = (
                (line.page == last_line.page and gap <= max(18, line.font_size * 1.5)
                 and line.bbox.x0 >= current["base_x"] + 4)
                or (line.page == last_line.page + 1 and line.bbox.top < 125)
            )
            if is_continuation and not _is_standalone_bold_label(line):
                item = current["items"][-1]
                item["text"] = join_text(item["text"], line.normalized_text)
                item["sources"].append(_source(line))
                current["last_line"] = line
                continue

        if not current or current_owner != owner or current["kind"] != "paragraph":
            flush()
            current_owner = owner
            current = {
                "kind": "paragraph", "text": line.normalized_text,
                "sources": [_source(line)], "last_line": line, "first_line": line,
            }
            continue

        last_line = current["last_line"]
        gap = line.bbox.top - last_line.bbox.bottom if line.page == last_line.page else 0
        continuous = (
            (line.page == last_line.page and gap <= max(18, line.font_size * 1.55))
            or (line.page == last_line.page + 1 and line.bbox.top < 125)
        )
        if continuous and not _is_standalone_bold_label(current["first_line"]):
            current["text"] = join_text(current["text"], line.normalized_text)
            current["sources"].append(_source(line))
            current["last_line"] = line
        else:
            flush()
            current_owner = owner
            current = {
                "kind": "paragraph", "text": line.normalized_text,
                "sources": [_source(line)], "last_line": line, "first_line": line,
            }

    flush()
    _update_page_ranges(sections)
    line_index = {line.id: line for page in pages for line in page.lines}
    for section in sections:
        section.elements.sort(key=lambda element: (element.page_start, element.sources[0].bbox.top, element.sequence))
        section.elements = _group_local_numbered_elements(section, line_index)
        section.elements = _group_labelled_blocks(section.elements, line_index, section.section_id)
    return sections


def _is_standalone_bold_label(line: PhysicalLine) -> bool:
    words = line.normalized_text.rstrip(":").split()
    punctuation_probe = line.normalized_text.rstrip('"\u201d\u2019) ]')
    return (
        "bold" in line.font_name.lower()
        and 1 <= len(words) <= 5
        and bool(line.normalized_text[:1].isupper())
        and not punctuation_probe.endswith((".", "?", "!"))
    )


def _is_title_like_label(text: str) -> bool:
    """Accept concise heading-like labels, not sentence-style introductions."""
    minor_words = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not words:
        return False
    return all(word.lower() in minor_words or word[0].isupper() for word in words)


def _is_local_group_candidate(element: Element) -> tuple[int, str] | None:
    if element.type != "procedure":
        return None
    items = element.data.get("items", [])
    if len(items) != 1:
        return None
    item = items[0]
    label = str(item.get("text", "")).strip()
    marker = str(item.get("marker", ""))
    if not marker.isdigit() or not label or len(label.split()) > 7 or len(label) > 80:
        return None
    first_word = re.sub(r"[^A-Za-z-]", "", label.split()[0]).lower()
    if first_word in IMPERATIVE_VERBS or label.endswith((".", ":", ";")):
        return None
    return int(marker), label


def _group_local_numbered_elements(
    section: Section, line_index: dict[str, PhysicalLine]
) -> list[Element]:
    candidates = [
        (index, candidate)
        for index, element in enumerate(section.elements)
        if (candidate := _is_local_group_candidate(element)) is not None
    ]
    valid_indices: set[int] = set()
    run: list[tuple[int, tuple[int, str]]] = []
    for candidate in candidates:
        if not run or candidate[1][0] == run[-1][1][0] + 1:
            run.append(candidate)
        else:
            if len(run) >= 2:
                valid_indices.update(index for index, _ in run)
            run = [candidate]
    if len(run) >= 2:
        valid_indices.update(index for index, _ in run)
    if not valid_indices:
        return section.elements

    result: list[Element] = []
    index = 0
    ordered_candidates = sorted(valid_indices)
    next_by_index = {
        candidate_index: (
            ordered_candidates[position + 1]
            if position + 1 < len(ordered_candidates)
            else len(section.elements)
        )
        for position, candidate_index in enumerate(ordered_candidates)
    }
    while index < len(section.elements):
        if index not in valid_indices:
            result.append(section.elements[index])
            index += 1
            continue
        marker, label = _is_local_group_candidate(section.elements[index]) or (0, "")
        label_element = section.elements[index]
        end = next_by_index[index]
        children = section.elements[index + 1 : end]
        group_id = f"section-{section.section_id}-local-group-{marker:02d}"
        _set_local_group_owner(children, group_id)
        pages = [label_element.page_start, label_element.page_end]
        pages.extend(page for child in children for page in (child.page_start, child.page_end))
        result.append(
            Element(
                id=group_id,
                type="local_group",
                sequence=label_element.sequence,
                text=label,
                page_start=min(pages),
                page_end=max(pages),
                sources=label_element.sources,
                data={
                    "label": label,
                    "order": marker,
                    "marker": str(marker),
                    "group_kind": "local_numbered_group",
                },
                children=children,
            )
        )
        index = end
    return result


def _set_local_group_owner(elements: list[Element], group_id: str) -> None:
    for element in elements:
        element.data["local_group_id"] = group_id
        _set_local_group_owner(element.children, group_id)


def _label_candidate(
    element: Element, line_index: dict[str, PhysicalLine]
) -> tuple[str, str, str] | None:
    if element.type != "paragraph" or not element.text:
        return None
    match = re.match(r"^([^:\n]{1,60}):(?:\s*(.*))?$", element.text, re.DOTALL)
    if (
        match
        and len(match.group(1).split()) <= 6
        and _is_title_like_label(match.group(1).strip())
    ):
        return match.group(1).strip(), (match.group(2) or "").strip(), "colon"
    if not element.sources or not element.sources[0].source_ids:
        return None
    line = line_index.get(element.sources[0].source_ids[0])
    if line and _is_standalone_bold_label(line) and len(element.text.split()) <= 5:
        return element.text.strip().rstrip(":"), "", "bold"
    return None


def _group_labelled_blocks(
    elements: list[Element], line_index: dict[str, PhysicalLine], owner_id: str
) -> list[Element]:
    for element in elements:
        if element.children:
            element.children = _group_labelled_blocks(element.children, line_index, element.id)

    candidates = {
        index: candidate
        for index, element in enumerate(elements)
        if (candidate := _label_candidate(element, line_index)) is not None
    }
    if not candidates:
        return elements

    result: list[Element] = []
    indices = sorted(candidates)
    next_by_index = {
        value: indices[position + 1] if position + 1 < len(indices) else len(elements)
        for position, value in enumerate(indices)
    }
    index = 0
    while index < len(elements):
        if index not in candidates:
            result.append(elements[index])
            index += 1
            continue
        label, inline_content, label_kind = candidates[index]
        original = elements[index]
        end = next_by_index[index]
        children: list[Element] = []
        if inline_content:
            children.append(
                Element(
                    id=f"{original.id}-content",
                    type="paragraph",
                    sequence=original.sequence,
                    text=inline_content,
                    page_start=original.page_start,
                    page_end=original.page_end,
                    sources=original.sources,
                    data={},
                )
            )
        children.extend(elements[index + 1 : end])
        group_id = f"{original.id}-labelled"
        pages = [original.page_start, original.page_end]
        pages.extend(page for child in children for page in (child.page_start, child.page_end))
        result.append(
            Element(
                id=group_id,
                type="labelled_block",
                sequence=original.sequence,
                text=label,
                page_start=min(pages),
                page_end=max(pages),
                sources=original.sources[:1],
                data={"label": label, "label_kind": label_kind, "owner_id": owner_id},
                children=children,
            )
        )
        index = end
    return result


def _nearest_existing_parent(parts: list[str], by_id: dict[str, Section]) -> str | None:
    for end in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in by_id:
            return candidate
    return None


def _owner_for_position(
    positions: list[tuple[tuple[int, float], Section]], page: int, top: float
) -> Section | None:
    owner = None
    for position, section in positions:
        if position <= (page, top):
            owner = section
        else:
            break
    return owner


def _materialize(state: dict[str, Any], sequence: int, owner: Section) -> Element:
    kind = state["kind"]
    if kind == "paragraph":
        sources = state["sources"]
        text = state["text"]
        data: dict[str, Any] = {}
    else:
        items = state["items"]
        sources = [source for item in items for source in item["sources"]]
        text = "\n".join(item["text"] for item in items)
        data = {
            "ordered": kind == "procedure",
            "items": [
                {
                    "text": item["text"],
                    "level": item["level"],
                    "marker": item["marker"],
                    "sources": [asdict(source) for source in item["sources"]],
                }
                for item in items
            ],
        }
    return Element(
        id=f"section-{owner.section_id}-element-{len(owner.elements) + 1:03d}",
        type=kind,
        sequence=sequence,
        text=text,
        page_start=min(source.page for source in sources),
        page_end=max(source.page for source in sources),
        sources=sources,
        data=data,
    )


def _table_element(table: dict[str, Any], sequence: int, owner: Section) -> Element:
    sources = [
        SourceSpan(
            page=int(source["page"]),
            bbox=BoundingBox.from_values(
                (
                    source["bbox"]["x0"], source["bbox"]["top"],
                    source["bbox"]["x1"], source["bbox"]["bottom"],
                )
            ),
            source_ids=list(source["source_ids"]),
        )
        for source in table["sources"]
    ]
    data = {key: value for key, value in table.items() if key not in {"sources", "page_start", "page_end"}}
    return Element(
        id=table["id"], type="table", sequence=sequence, text=table.get("caption"),
        page_start=table["page_start"], page_end=table["page_end"], sources=sources, data=data,
    )


def _figure_element(
    figure: dict[str, Any], sequence: int, owner: Section, pages: list[PhysicalPage]
) -> Element:
    page = pages[figure["page"] - 1]
    caption_line = next(line for line in page.lines if line.id == figure["caption_line_id"])
    data = {key: value for key, value in figure.items() if key not in {"caption_line_id", "page"}}
    return Element(
        id=figure["id"], type="figure", sequence=sequence, text=figure["caption"],
        page_start=figure["page"], page_end=figure["page"], sources=[_source(caption_line)], data=data,
    )


def _update_page_ranges(sections: list[Section]) -> None:
    by_id = {section.section_id: section for section in sections}
    for section in reversed(sections):
        own_pages = [page for element in section.elements for page in (element.page_start, element.page_end)]
        if own_pages:
            section.page_end = max(section.page_end, *own_pages)
        if section.parent_section_id and section.parent_section_id in by_id:
            parent = by_id[section.parent_section_id]
            parent.page_end = max(parent.page_end, section.page_end)
