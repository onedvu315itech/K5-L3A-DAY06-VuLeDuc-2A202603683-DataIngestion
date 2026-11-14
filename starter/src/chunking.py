
"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage

import re


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""

    if size <= 0:
        raise ValueError("size must be greater than 0")

    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    text = page.text

    if not text:
        return []

    chunks = []
    step = size - overlap

    for index, start in enumerate(range(0, len(text), step)):
        chunk_text = text[start:start + size]

        if not chunk_text.strip():
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-chunk{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=page.metadata.copy()
            )
        )

        if start + size >= len(text):
            break

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Structure-aware chunking: split at headings, pack paragraphs under them."""

    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    headings = set(page.metadata.get("headings", []))

    section = page.metadata.get("section_at_start")
    section_id = page.metadata.get("section_id_at_start")

    chunks = []
    current_text = ""
    snap = (section, section_id)  # section state when the current chunk began

    def add_chunk(chunk_text: str, sec, sec_id) -> None:
        meta = page.metadata.copy()
        for key in ("headings", "section_at_start", "section_id_at_start"):
            meta.pop(key, None)
        if sec:
            meta["section"] = sec
        if sec_id:
            meta["section_id"] = sec_id

        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-chunk{len(chunks)}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=meta,
            )
        )

    for paragraph in paragraphs:

        # A heading closes the current chunk and starts a new one.
        if paragraph in headings:
            if current_text:
                add_chunk(current_text, *snap)
            section = paragraph
            m = re.match(r"^(\d+\.\d+)\b", paragraph)
            if m:
                section_id = m.group(1)
            current_text = paragraph
            snap = (section, section_id)
            continue

        if not current_text:
            snap = (section, section_id)

        if len(paragraph) > max_size:
            if current_text:
                add_chunk(current_text, *snap)
                current_text = ""
            for start in range(0, len(paragraph), max_size):
                add_chunk(paragraph[start:start + max_size], section, section_id)
            continue

        candidate = f"{current_text}\n\n{paragraph}" if current_text else paragraph

        if len(candidate) <= max_size:
            current_text = candidate
        else:
            add_chunk(current_text, *snap)
            current_text = paragraph
            snap = (section, section_id)

    # Skip a chunk that is only a trailing heading; the next page inherits it.
    if current_text and current_text not in headings:
        add_chunk(current_text, *snap)

    return chunks
