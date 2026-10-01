"""End-to-end native parser for the TESTRX User Manual."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pdfplumber

from .. import __version__
from .domain import BoundingBox, CanonicalDocument, PhysicalLine, PhysicalPage
from .figures import classify_figure_captions, extract_figure_regions
from .inspection import collect_parsing_warnings, render_semantic_structure
from .normalize import normalize_text
from .structure import build_sections, classify_headings
from .tables import extract_table_fragments, group_logical_tables


SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_manual(path: str | Path) -> CanonicalDocument:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    pages: list[PhysicalPage] = []
    with pdfplumber.open(source) as pdf:
        source_profile = _source_profile(pdf.pages)
        metadata = _metadata(pdf.metadata or {}, len(pdf.pages), source_profile)
        for page_number, pdf_page in enumerate(pdf.pages, start=1):
            page_type = "cover" if page_number == 1 else "toc" if 2 <= page_number <= 4 else "body"
            lines = _extract_lines(pdf_page, page_number, page_type)
            table_fragments = extract_table_fragments(pdf_page, page_number, lines)
            figure_regions = extract_figure_regions(pdf_page, page_number)
            classify_figure_captions(lines)
            pages.append(
                PhysicalPage(
                    page_number=page_number,
                    width=round(float(pdf_page.width), 3),
                    height=round(float(pdf_page.height), 3),
                    page_type=page_type,
                    lines=lines,
                    table_fragments=table_fragments,
                    figure_regions=figure_regions,
                )
            )

    classify_headings(pages)
    logical_tables = group_logical_tables(pages)
    sections = build_sections(pages, logical_tables)
    return CanonicalDocument(
        schema_version=SCHEMA_VERSION,
        parser_version=__version__,
        document_id="testrx_user_manual",
        title="TESTRX User Manual",
        source_file=source.name,
        source_sha256=sha256_file(source),
        metadata=metadata,
        pages=pages,
        sections=sections,
    )


def _extract_lines(pdf_page, page_number: int, page_type: str) -> list[PhysicalLine]:
    extracted = pdf_page.dedupe_chars().extract_text_lines(return_chars=True, strip=True)
    lines: list[PhysicalLine] = []
    for order, item in enumerate(extracted, start=1):
        text = item.get("text", "").strip()
        if not text:
            continue
        chars = [char for char in item.get("chars", []) if char.get("text", "").strip()]
        font_counts = Counter(char.get("fontname", "unknown") for char in chars)
        font_name = font_counts.most_common(1)[0][0] if font_counts else "unknown"
        font_size = max((float(char.get("size", 0.0)) for char in chars), default=0.0)
        classification = _initial_classification(text, float(item["top"]), page_type)
        lines.append(
            PhysicalLine(
                id=f"p{page_number:03d}-line-{order:03d}",
                page=page_number,
                order=order,
                text=text,
                normalized_text=normalize_text(text),
                bbox=BoundingBox.from_values(
                    (item["x0"], item["top"], item["x1"], item["bottom"])
                ),
                font_name=font_name,
                font_size=round(font_size, 3),
                classification=classification,
            )
        )
    return lines


def _initial_classification(text: str, top: float, page_type: str) -> str:
    normalized = normalize_text(text)
    if top >= 770 or "All rights reserved" in normalized or "No passing on to third parties" in normalized:
        return "boilerplate"
    if page_type == "cover":
        return "cover"
    if page_type == "toc":
        return "toc"
    return "body"


def _source_profile(pdf_pages) -> dict[str, Any]:
    character_counts = [len(page.chars) for page in pdf_pages]
    native_pages = sum(count >= 20 for count in character_counts)
    page_count = len(character_counts)
    native_ratio = native_pages / page_count if page_count else 0.0
    total_characters = sum(character_counts)
    source_type = (
        "native_pdf"
        if page_count and native_ratio >= 0.8 and total_characters >= page_count * 100
        else "unsupported_non_native_pdf"
    )
    return {
        "source_type": source_type,
        "native_text_pages": native_pages,
        "page_native_ratio": round(native_ratio, 4),
        "native_character_count": total_characters,
        "classification_basis": "pages_with_at_least_20_native_characters",
    }


def _metadata(
    raw: dict[str, Any], page_count: int, source_profile: dict[str, Any]
) -> dict[str, Any]:
    keep = {key.lstrip("/"): normalize_text(str(value)) for key, value in raw.items() if value is not None}
    keep.update(
        {
            "page_count": page_count,
            "language": "en",
            "document_type": "user_manual",
            "toc_pages": [2, 3, 4],
            "ocr_used": False,
        }
    )
    keep.update(source_profile)
    return keep


def write_outputs(document: CanonicalDocument, output_dir: str | Path) -> tuple[dict, dict]:
    from .validation import validate_document

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    document_path = destination / "document.json"
    semantic_path = destination / "semantic_structure.md"
    validation_path = destination / "validation_report.json"
    warnings_path = destination / "parsing_warnings.json"
    inventory_path = destination / "page_inventory.csv"

    document_path.write_text(
        json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    warnings = collect_parsing_warnings(document)
    report = validate_document(document, warnings)
    validation_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    warnings_path.write_text(
        json.dumps(warnings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    semantic_path.write_text(render_semantic_structure(document), encoding="utf-8")
    _write_inventory(document, inventory_path, warnings)
    return report, warnings


def _write_inventory(document: CanonicalDocument, path: Path, warnings: dict) -> None:
    section_pages: dict[int, list[str]] = {}
    for section in document.sections:
        for page_number in range(section.page_start, section.page_end + 1):
            section_pages.setdefault(page_number, []).append(section.section_id)
    warnings_by_page: dict[int, list[str]] = {}
    for warning in warnings["items"]:
        for page_number in warning.get("pages", []):
            warnings_by_page.setdefault(page_number, []).append(warning["type"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "page", "text_line_count", "section_ids", "table_count",
                "figure_count", "caption_count", "page_type", "warnings",
            ],
        )
        writer.writeheader()
        for page in document.pages:
            writer.writerow(
                {
                    "page": page.page_number,
                    "text_line_count": len(page.lines),
                    "section_ids": "|".join(section_pages.get(page.page_number, [])),
                    "table_count": len(page.table_fragments),
                    "figure_count": len(page.figure_regions),
                    "caption_count": sum(line.classification == "figure_caption" for line in page.lines),
                    "page_type": page.page_type,
                    "warnings": "|".join(sorted(set(warnings_by_page.get(page.page_number, [])))),
                }
            )
