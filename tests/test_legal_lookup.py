import csv

from src.legal_lookup import _load_rows, lookup_exact_sections


def test_exact_section_prefers_matching_act(tmp_path, monkeypatch):
    csv_path = tmp_path / "laws.csv"
    fieldnames = [
        "law_title",
        "section_description",
        "section_chapter_name",
        "section_name",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "law_title": "Another Act, 2000",
                    "section_description": "379. An unrelated provision.",
                    "section_chapter_name": "Other",
                    "section_name": "Other rule",
                },
                {
                    "law_title": "The Penal Code, 1860",
                    "section_description": (
                        "379. Whoever commits theft may be imprisoned for three "
                        "years, or fined, or both."
                    ),
                    "section_chapter_name": "Offences Against Property",
                    "section_name": "Punishment for theft",
                },
            ]
        )

    monkeypatch.setenv("LEGAL_CSV_PATH", str(csv_path))
    _load_rows.cache_clear()
    documents = lookup_exact_sections(
        "What is the punishment under Section 379 of the Penal Code, 1860?"
    )

    assert documents[0].metadata["law_title"] == "The Penal Code, 1860"
    assert "three years" in documents[0].page_content


def test_query_without_section_uses_semantic_retrieval_only():
    assert lookup_exact_sections("What is theft?") == []
