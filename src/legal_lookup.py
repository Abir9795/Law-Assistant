"""Deterministic lookup for queries that name an exact statutory section."""

import csv
import os
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

SECTION_PATTERN = re.compile(r"\b(?:section|sec\.?)[\s:-]*(\d+[A-Za-z]?)\b", re.I)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _words(value: str) -> set[str]:
    return {word.lower() for word in WORD_PATTERN.findall(value) if len(word) > 2}


@lru_cache(maxsize=4)
def _load_rows(csv_path: str) -> tuple[dict[str, str], ...]:
    path = Path(csv_path)
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return tuple(csv.DictReader(handle))


def lookup_exact_sections(query: str, limit: int = 2) -> list[Document]:
    """Return locally stored provisions when a query explicitly names a section."""
    match = SECTION_PATTERN.search(query)
    if not match:
        return []

    section = match.group(1).lower()
    csv_path = os.getenv("LEGAL_CSV_PATH", "Data/bdlaws_ds_formatted.csv")
    query_words = _words(query)
    candidates = []

    for row_number, row in enumerate(_load_rows(csv_path), start=2):
        description = (row.get("section_description") or "").strip()
        if not re.match(rf"^\s*\[?\s*{re.escape(section)}(?:\.|\s)", description, re.I):
            continue
        title = (row.get("law_title") or "Unknown law").strip()
        title_score = len(_words(title) & query_words)
        candidates.append((title_score, row_number, title, description, row))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    documents = []
    for _, row_number, title, description, row in candidates[:limit]:
        chapter = (row.get("section_chapter_name") or "").strip()
        section_name = (row.get("section_name") or "").strip()
        content = "\n".join(
            part
            for part in (
                f"Act: {title}",
                f"Chapter: {chapter}" if chapter else "",
                f"Provision: {description}",
                f"Heading: {section_name}" if section_name else "",
            )
            if part
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": csv_path,
                    "row": row_number,
                    "section": section,
                    "law_title": title,
                    "retrieval": "exact-section",
                },
            )
        )
    return documents
