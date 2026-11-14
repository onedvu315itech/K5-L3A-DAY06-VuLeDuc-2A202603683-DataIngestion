
"""Read and normalize source data."""
from pathlib import Path
import re
import unicodedata

import fitz

from .models import DocumentPage
from collections import Counter


def _extract_blocks(pdf_page) -> list[dict]:
    """Return text blocks with font size, boldness and vertical position."""
    blocks = []
    page_height = pdf_page.rect.height
    data = pdf_page.get_text("dict", sort=True)

    for block in data["blocks"]:
        if block.get("type") != 0:  # skip images
            continue

        parts, sizes, bolds = [], [], []
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if line_text:
                parts.append(line_text)
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(span["size"])
                    bolds.append(bool(span["flags"] & 16))

        text = normalize_text(" ".join(parts))
        if not text or not sizes:
            continue

        y0, y1 = block["bbox"][1], block["bbox"][3]
        blocks.append({
            "text": text,
            "size": max(sizes),
            "bold": all(bolds),
            "in_margin": y1 < page_height * 0.07 or y0 > page_height * 0.93,
        })

    return blocks


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical PDF page, keeping paragraph and heading structure."""

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    with fitz.open(path) as pdf:
        raw_pages = [_extract_blocks(p) for p in pdf]

    # Body font size = the size that covers the most characters in the document.
    size_chars = Counter()
    for blocks in raw_pages:
        for b in blocks:
            size_chars[round(b["size"], 1)] += len(b["text"])
    body_size = size_chars.most_common(1)[0][0] if size_chars else 0

    def is_heading(b: dict) -> bool:
        text = b["text"]
        if b["in_margin"] or text.isdigit() or len(text) > 120:
            return False
        if b["size"] >= body_size * 1.1:
            return True
        return b["bold"] and not text.endswith(".")

    pages: list[DocumentPage] = []
    current_section = None
    current_section_id = None

    for page_index, blocks in enumerate(raw_pages):
        text = normalize_text("\n\n".join(b["text"] for b in blocks))
        headings = [b["text"] for b in blocks if is_heading(b)]

        metadata = {
            "is_empty": not bool(text.strip()),
            "physical_page": page_index + 1,
            "needs_ocr": not bool(text.strip()),
            "headings": headings,
            "section_at_start": current_section,
            "section_id_at_start": current_section_id,
        }

        # Carry the latest heading into the next page.
        for h in headings:
            current_section = h
            m = re.match(r"^(\d+\.\d+)\b", h)
            if m:
                current_section_id = m.group(1)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_index + 1,
                text=text,
                metadata=metadata,
            )
        )

    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph structure."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text:
        return ""

    # 1. Normalize Unicode characters (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Replace non-breaking spaces with regular spaces
    text = text.replace("\u00A0", " ")

    # 4. Remove unwanted control characters, preserving tabs and newlines
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # 5. Normalize horizontal whitespace
    text = re.sub(r"[^\S\n]+", " ", text)

    # 6. Remove spaces around line boundaries
    lines = [
        line.strip()
        for line in text.split("\n")
    ]

    # 7. Preserve paragraph boundaries but remove excessive blank lines
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
