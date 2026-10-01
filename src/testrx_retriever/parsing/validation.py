"""Whole-document and source-specific parsing validation."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .domain import CanonicalDocument, Element


EXPECTED_SHA256 = "9CC50B1955AB92884CB0C1AA1A1B6ED437BDBD419962A2FDC6818B4D17073B99"
EXPECTED_SECTIONS = {
    "1", "4", "12", "12.1", "12.2", "12.3", "12.5", "12.5.1", "12.5.2",
    "12.7", "14.6", "14.7.1", "16.2.2", "16.2.4.1", "17", "18",
}


def validate_document(document: CanonicalDocument, warnings: dict | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, message: str, evidence: Any = None, severity: str = "FAIL") -> None:
        checks.append(
            {
                "id": identifier,
                "status": "PASS" if passed else severity,
                "message": message,
                "evidence": evidence,
            }
        )

    check("source.sha256", document.source_sha256 == EXPECTED_SHA256, "Source checksum matches the approved manual.", document.source_sha256)
    check("source.supported", document.metadata.get("source_type") == "native_pdf", "Source has sufficient native text for this non-OCR parser.", {"source_type": document.metadata.get("source_type"), "native_text_pages": document.metadata.get("native_text_pages"), "page_native_ratio": document.metadata.get("page_native_ratio")})
    check("pages.count", len(document.pages) == 61, "All 61 physical pages are represented.", len(document.pages))
    check("pages.sequence", [page.page_number for page in document.pages] == list(range(1, 62)), "Page numbering is contiguous.")

    section_ids = [section.section_id for section in document.sections]
    check("sections.unique", len(section_ids) == len(set(section_ids)), "Section identifiers are unique.", len(section_ids))
    missing_sections = sorted(EXPECTED_SECTIONS - set(section_ids))
    check("sections.landmarks", not missing_sections, "Known hierarchy landmarks were detected.", {"missing": missing_sections})
    by_id = {section.section_id: section for section in document.sections}
    orphaned = [section.section_id for section in document.sections if section.parent_section_id and section.parent_section_id not in by_id]
    check("sections.parents", not orphaned, "Every declared parent section exists.", {"orphaned": orphaned})
    roots = {section.section_id for section in document.sections if section.parent_section_id is None}
    expected_roots = {str(value) for value in range(1, 19)}
    check("sections.roots", roots == expected_roots, "Top-level sections 1 through 18 are present.", {"missing": sorted(expected_roots - roots), "unexpected": sorted(roots - expected_roots)})
    invalid_ranges = [section.section_id for section in document.sections if section.page_end < section.page_start]
    check("sections.page_ranges", not invalid_ranges, "Section page ranges are valid.", {"invalid": invalid_ranges})

    elements = [element for section in document.sections for element in _walk_elements(section.elements)]
    element_ids = [element.id for element in elements]
    duplicate_element_ids = sorted(identifier for identifier, count in Counter(element_ids).items() if count > 1)
    check("elements.unique_ids", not duplicate_element_ids, "Canonical element identifiers are unique.", {"duplicates": duplicate_element_ids})
    missing_sources = [element.id for element in elements if not element.sources]
    check("elements.provenance", not missing_sources, "Every logical element has source provenance.", {"missing": missing_sources})
    element_types = Counter(element.type for element in elements)
    check("elements.types", all(element_types[kind] > 0 for kind in ("paragraph", "list", "procedure", "table", "figure", "local_group", "labelled_block")), "All required element families are present.", dict(element_types))

    local_groups = [element for element in elements if element.type == "local_group"]
    empty_groups = [element.id for element in local_groups if not element.children]
    check("local_groups.content", bool(local_groups) and not empty_groups, "Local semantic groups exist and own content.", {"count": len(local_groups), "empty": empty_groups})
    group_without_figure = [element.id for element in local_groups if not any(child.type == "figure" for child in _walk_elements(element.children))]
    check("local_groups.figures", not group_without_figure, "Each numbered local group owns its corresponding figure.", {"without_figure": group_without_figure})

    tables = [element for element in elements if element.type == "table"]
    table_ids = {element.data.get("printed_id") for element in tables}
    missing_tables = sorted(set(str(value) for value in range(1, 10)) - table_ids)
    check("tables.captioned", not missing_tables, "Printed Tables 1 through 9 were reconstructed.", {"found": sorted(value for value in table_ids if value), "missing": missing_tables})
    table_1 = next((element for element in tables if element.data.get("printed_id") == "1"), None)
    table_7 = next((element for element in tables if element.data.get("printed_id") == "7"), None)
    table_8 = next((element for element in tables if element.data.get("printed_id") == "8"), None)
    check("tables.table_1_continuation", bool(table_1 and table_1.page_start == 9 and table_1.page_end == 10), "Table 1 spans pages 9-10.", _range(table_1))
    check("tables.table_7_continuation", bool(table_7 and table_7.page_start == 29 and table_7.page_end == 30), "Table 7 spans pages 29-30.", _range(table_7))
    table_7_context = table_7.data.get("resolved_rows", []) if table_7 else []
    check("tables.table_7_context", bool(table_7_context and all(row[0] == "Action" for row in table_7_context)), "Table 7 merged Action context is propagated across both pages.", table_7_context)
    table_8_context = table_8.data.get("resolved_rows", []) if table_8 else []
    check("tables.table_8_context", bool(table_8_context and all(row[0] == "Entity Type" for row in table_8_context)), "Table 8 merged Entity Type context is propagated.", table_8_context)
    malformed_tables = [element.id for element in tables if len(element.data.get("columns", [])) < 2 or not element.data.get("resolved_rows")]
    check("tables.structure", not malformed_tables, "Every table has columns and semantic rows.", {"malformed": malformed_tables})

    figures = [element for element in elements if element.type == "figure"]
    associated = sum(bool(element.data.get("region_id")) for element in figures)
    check("figures.captions", len(figures) >= 45, "Native screenshot captions were retained.", len(figures), severity="WARN")
    check("figures.regions", associated >= max(1, int(len(figures) * 0.8)), "Most captions are associated with a large native image region.", {"captions": len(figures), "associated": associated}, severity="WARN")
    printed_figure_ids = Counter(element.data.get("printed_id") for element in figures)
    duplicate_printed_ids = {key: value for key, value in printed_figure_ids.items() if key and value > 1}
    check("figures.printed_id_duplicates", not duplicate_printed_ids, "Printed figure identifiers should be unique; duplicates are preserved with page-qualified canonical IDs.", duplicate_printed_ids, severity="WARN")

    cross_page = {section_id: _section_range(by_id.get(section_id)) for section_id in ("12.3", "14.6", "16.2.2")}
    cross_page_ok = all(value and value[1] > value[0] for value in cross_page.values())
    check("sections.cross_page", cross_page_ok, "Known cross-page sections retain continuity.", cross_page)

    boilerplate_pages = {
        page.page_number for page in document.pages if any(line.classification == "boilerplate" for line in page.lines)
    }
    check("boilerplate.detected", len(boilerplate_pages) >= 55, "Repeated footer/page-number evidence is classified.", len(boilerplate_pages), severity="WARN")
    check("toc.classified", [page.page_type for page in document.pages[1:4]] == ["toc", "toc", "toc"], "Pages 2-4 are structural TOC pages.")
    toc_ids = {
        match.group(1)
        for page in document.pages[1:4]
        for line in page.lines
        if (match := re.match(r"^(\d+(?:\.\d+)*)\.?\s", line.normalized_text))
    }
    body_ids = set(section_ids)
    check("toc.heading_coverage", toc_ids <= body_ids, "Every section listed in the TOC exists in the reconstructed body hierarchy.", {"missing_from_body": sorted(toc_ids - body_ids), "body_only": sorted(body_ids - toc_ids)})

    semantic_source_ids = {
        source_id for element in elements for source in element.sources for source_id in source.source_ids
    }
    unowned_body_lines = [
        line.id
        for page in document.pages
        for line in page.lines
        if line.classification == "body" and line.id not in semantic_source_ids
    ]
    check("coverage.semantic_lines", not unowned_body_lines, "Every eligible body line is owned by a logical element.", {"unowned": unowned_body_lines[:25], "count": len(unowned_body_lines)})
    if warnings is not None:
        check("warnings.errors", warnings["summary"]["errors"] == 0, "Parsing warnings contain no errors.", warnings["summary"])

    status_counts = Counter(item["status"] for item in checks)
    overall = "FAIL" if status_counts["FAIL"] else "PASS_WITH_WARNINGS" if status_counts["WARN"] else "PASS"
    return {
        "overall_status": overall,
        "summary": {
            "checks": len(checks),
            "passed": status_counts["PASS"],
            "warnings": status_counts["WARN"],
            "failed": status_counts["FAIL"],
            "pages": len(document.pages),
            "sections": len(document.sections),
            "elements": len(elements),
            "semantic_groups": element_types["local_group"] + element_types["labelled_block"],
            "element_types": dict(element_types),
        },
        "checks": checks,
    }


def _range(element) -> list[int] | None:
    return [element.page_start, element.page_end] if element else None


def _section_range(section) -> list[int] | None:
    return [section.page_start, section.page_end] if section else None


def _walk_elements(elements: list[Element]):
    for element in elements:
        yield element
        yield from _walk_elements(element.children)
