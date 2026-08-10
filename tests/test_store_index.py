from types import SimpleNamespace

from store_index import _chunk_id


def test_chunk_ids_are_stable_and_content_sensitive():
    first = SimpleNamespace(
        page_content="Section 1", metadata={"source": "law.pdf", "page": 1}
    )
    same = SimpleNamespace(
        page_content="Section 1", metadata={"source": "law.pdf", "page": 1}
    )
    different = SimpleNamespace(
        page_content="Section 2", metadata={"source": "law.pdf", "page": 1}
    )

    assert _chunk_id(first) == _chunk_id(same)
    assert _chunk_id(first) != _chunk_id(different)
