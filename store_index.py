"""Create (if necessary) and populate the Pinecone vector index."""

import argparse
import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_hugging_face_embeddings,
    load_all_documents,
    split_into_chunks,
)

EMBEDDING_DIMENSION = 384


def _chunk_id(chunk) -> str:
    source = chunk.metadata.get("source", "unknown")
    page = chunk.metadata.get("page", "")
    value = f"{source}:{page}:{chunk.page_content}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def ensure_index(client: Pinecone, index_name: str) -> None:
    existing = {index.name for index in client.list_indexes()}
    if index_name not in existing:
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )

    while True:
        status = client.describe_index(index_name).status
        ready = (
            status.get("ready", False)
            if isinstance(status, dict)
            else getattr(status, "ready", False)
        )
        if ready:
            break
        print(f"Waiting for Pinecone index '{index_name}'...")
        time.sleep(2)


def build_index(data_dir: Path, batch_size: int) -> None:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env first.")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    index_name = os.getenv("PINECONE_INDEX", "lawbot")
    client = Pinecone(api_key=api_key)
    ensure_index(client, index_name)

    documents = load_all_documents(str(data_dir))
    chunks = split_into_chunks(documents)
    embeddings = download_hugging_face_embeddings()
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )

    print(f"Uploading {len(chunks)} chunks to '{index_name}'...")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vector_store.add_documents(batch, ids=[_chunk_id(chunk) for chunk in batch])
        print(f"Uploaded {min(start + batch_size, len(chunks))}/{len(chunks)}")
    print("Indexing complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    arguments = parse_args()
    build_index(arguments.data_dir, arguments.batch_size)
