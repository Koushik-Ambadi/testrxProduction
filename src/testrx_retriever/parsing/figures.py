"""Figure caption and native image-region handling."""

from __future__ import annotations

import re

from .domain import BoundingBox, FigureRegion, PhysicalLine


FIGURE_CAPTION_RE = re.compile(r"^Snp\.?\s*(\d+)\s*[-–—]\s*(.+)$", re.IGNORECASE)


def extract_figure_regions(pdf_page, page_number: int) -> list[FigureRegion]:
    regions: list[FigureRegion] = []
    for index, image in enumerate(pdf_page.images, start=1):
        bbox = BoundingBox.from_values(
            (image["x0"], image["top"], image["x1"], image["bottom"])
        )
        width = bbox.x1 - bbox.x0
        height = bbox.bottom - bbox.top
        if width >= 140 and height >= 70 and bbox.top < pdf_page.height - 90:
            regions.append(FigureRegion(f"p{page_number:03d}-image-{index:02d}", page_number, bbox))
    return regions


def classify_figure_captions(lines: list[PhysicalLine]) -> list[PhysicalLine]:
    captions: list[PhysicalLine] = []
    for line in lines:
        if FIGURE_CAPTION_RE.match(line.normalized_text):
            line.classification = "figure_caption"
            captions.append(line)
    return captions


def figure_from_caption(
    line: PhysicalLine, regions: list[FigureRegion]
) -> dict:
    match = FIGURE_CAPTION_RE.match(line.normalized_text)
    assert match is not None
    printed_id = str(int(match.group(1)))
    above = [region for region in regions if region.bbox.bottom <= line.bbox.top + 18]
    region = min(above, key=lambda item: line.bbox.top - item.bbox.bottom, default=None)
    return {
        "id": f"snp_{printed_id}_p{line.page:03d}",
        "printed_id": printed_id,
        "caption": match.group(2).strip(),
        "page": line.page,
        "region_id": region.id if region else None,
        "region_bbox": region.bbox if region else None,
        "semantically_interpreted": False,
        "caption_line_id": line.id,
    }
