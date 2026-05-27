from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
from typing import Iterable

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import (
    DEFAULT_CHUNKED_OUTPUT_PATH,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_jsonl(input_path: str | Path) -> list[dict]:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    records = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_path}"
                ) from error

    return records


def batch_records(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start_index in range(0, len(records), batch_size):
        yield records[start_index:start_index + batch_size]


def sanitize_metadata_value(value):
    """
    Chroma metadata values should be simple scalar values.

    We convert unsupported values into strings.
    """
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def build_metadata(chunk: dict) -> dict:
    """
    Metadata saved with each chunk.

    Keep metadata small and filter-friendly.
    """
    metadata_fields = [
        "document_id",
        "paper_title",
        "file_name",
        "file_path",
        "page_number",
        "section_title",
        "chunk_index",
        "word_count",
        "source",
    ]

    metadata = {}

    for field in metadata_fields:
        metadata[field] = sanitize_metadata_value(chunk.get(field))

    return metadata


class ChromaVectorStore:
    """
    Local ChromaDB vector store for PaperMind chunks.

    Chroma stores:
    - ids
    - text documents
    - embeddings
    - metadata

    We use PersistentClient so the database is saved locally under data/vector_store/chroma.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PaperMind research paper chunks",
            },
        )

    def reset_collection(self):
        """
        Delete and recreate the collection.

        Use this when rebuilding embeddings from scratch.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info("Deleted existing collection: %s", self.collection_name)
        except Exception:
            logger.info("No existing collection to delete: %s", self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PaperMind research paper chunks",
            },
        )

    def upsert_chunks(
        self,
        chunks: list[dict],
        embedder: SentenceTransformerEmbedder,
        embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
        upsert_batch_size: int = 128,
    ):
        if not chunks:
            raise ValueError("No chunks provided for upsert.")

        logger.info("Creating embeddings and storing %s chunks", len(chunks))

        for batch_number, chunk_batch in enumerate(
            batch_records(chunks, upsert_batch_size),
            start=1,
        ):
            ids = [chunk["chunk_id"] for chunk in chunk_batch]
            documents = [chunk["text"] for chunk in chunk_batch]
            metadatas = [build_metadata(chunk) for chunk in chunk_batch]

            embeddings = embedder.embed_texts(
                texts=documents,
                batch_size=embedding_batch_size,
                normalize_embeddings=True,
            )

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(
                "Upserted batch %s | chunks=%s",
                batch_number,
                len(chunk_batch),
            )

        logger.info("Vector store upsert complete")

    def count(self) -> int:
        return self.collection.count()

    def query(
        self,
        query_text: str,
        embedder: SentenceTransformerEmbedder,
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        query_embedding = embedder.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return results


def build_vector_store_from_chunks(
    chunks_path: str | Path = DEFAULT_CHUNKED_OUTPUT_PATH,
    persist_dir: str | Path = CHROMA_DB_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    reset: bool = False,
):
    chunks = load_jsonl(chunks_path)

    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_path}")

    logger.info("Loaded %s chunks from %s", len(chunks), chunks_path)

    embedder = SentenceTransformerEmbedder(
        model_name=embedding_model_name,
    )

    vector_store = ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    if reset:
        vector_store.reset_collection()

    vector_store.upsert_chunks(
        chunks=chunks,
        embedder=embedder,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
    )

    logger.info("Collection count: %s", vector_store.count())

    return vector_store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector store from PaperMind chunks."
    )

    parser.add_argument(
        "--chunks-path",
        type=str,
        default=str(DEFAULT_CHUNKED_OUTPUT_PATH),
        help="Path to paper chunks JSONL.",
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(CHROMA_DB_DIR),
        help="Directory where ChromaDB should persist data.",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=CHROMA_COLLECTION_NAME,
        help="Chroma collection name.",
    )

    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="SentenceTransformer embedding model name.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before rebuilding.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting vector store build")
    logger.info("Chunks path: %s", args.chunks_path)
    logger.info("Persist dir: %s", args.persist_dir)
    logger.info("Collection name: %s", args.collection_name)
    logger.info("Embedding model: %s", args.embedding_model_name)

    build_vector_store_from_chunks(
        chunks_path=args.chunks_path,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model_name=args.embedding_model_name,
        reset=args.reset,
    )

    logger.info("Vector store build complete")


if __name__ == "__main__":
    main()