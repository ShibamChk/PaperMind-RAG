from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import argparse
import json
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from src.embeddings.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """
    Clean citation-ready retrieval result.

    This structure hides raw vector database output and gives the generation
    module exactly what it needs:
    - retrieved text
    - paper metadata
    - source citation
    - rank and relevance score
    """

    rank: int
    chunk_id: str
    text: str
    paper_title: str
    file_name: str
    page_number: int
    section_title: str
    source: str
    distance: float
    relevance_score: float

    def to_dict(self) -> dict:
        return asdict(self)


class PaperRetriever:
    """
    Citation-ready retriever for PaperMind.

    It retrieves chunks from ChromaDB and formats them into structured results
    that can later be passed into an LLM answer generator.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self.embedder = SentenceTransformerEmbedder(
            model_name=embedding_model_name,
        )

        self.vector_store = ChromaVectorStore(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )

        collection_count = self.vector_store.count()

        if collection_count == 0:
            logger.warning(
                "Vector store collection is empty. Build the vector store first."
            )
        else:
            logger.info("Vector store loaded with %s chunks", collection_count)

    def search(
        self,
        query: str,
        top_k: int = 5,
        paper_title: Optional[str] = None,
        file_name: Optional[str] = None,
        section_title: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Search for relevant chunks.

        Optional filters:
        - paper_title
        - file_name
        - section_title

        These are exact metadata filters. For flexible semantic filtering,
        use the query text itself.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        where_filter = self._build_where_filter(
            paper_title=paper_title,
            file_name=file_name,
            section_title=section_title,
        )

        raw_results = self.vector_store.query(
            query_text=query,
            embedder=self.embedder,
            top_k=top_k,
            where=where_filter,
        )

        retrieved_chunks = self._convert_chroma_results(raw_results)

        return retrieved_chunks

    def format_context_for_llm(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """
        Convert retrieved chunks into a clean context block for an LLM.

        This format makes citation grounding easier in Step 6.
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context_blocks = []

        for chunk in retrieved_chunks:
            block = (
                f"[Source {chunk.rank}]\n"
                f"Paper: {chunk.paper_title}\n"
                f"File: {chunk.file_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Section: {chunk.section_title}\n"
                f"Citation: {chunk.source}\n"
                f"Relevance Score: {chunk.relevance_score:.4f}\n"
                f"Text:\n{chunk.text}"
            )

            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def search_as_context(
        self,
        query: str,
        top_k: int = 5,
        paper_title: Optional[str] = None,
        file_name: Optional[str] = None,
        section_title: Optional[str] = None,
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Convenience method:
        return both structured chunks and formatted LLM context.
        """
        chunks = self.search(
            query=query,
            top_k=top_k,
            paper_title=paper_title,
            file_name=file_name,
            section_title=section_title,
        )

        context = self.format_context_for_llm(chunks)

        return chunks, context

    @staticmethod
    def _build_where_filter(
        paper_title: Optional[str] = None,
        file_name: Optional[str] = None,
        section_title: Optional[str] = None,
    ) -> Optional[dict]:
        filters = []

        if paper_title:
            filters.append({"paper_title": paper_title})

        if file_name:
            filters.append({"file_name": file_name})

        if section_title:
            filters.append({"section_title": section_title})

        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    @staticmethod
    def _distance_to_relevance_score(distance: float) -> float:
        """
        Convert vector distance into a human-readable relevance score.

        Chroma returns distance where lower is better.
        This transformation gives a bounded score:
            high score = more relevant
            low score = less relevant
        """
        return 1.0 / (1.0 + float(distance))

    def _convert_chroma_results(self, raw_results: dict) -> list[RetrievedChunk]:
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        retrieved_chunks = []

        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 999.0
            chunk_id = ids[index] if index < len(ids) else f"unknown_{index}"

            retrieved_chunk = RetrievedChunk(
                rank=index + 1,
                chunk_id=str(chunk_id),
                text=document,
                paper_title=str(metadata.get("paper_title", "Unknown")),
                file_name=str(metadata.get("file_name", "Unknown")),
                page_number=int(metadata.get("page_number", -1)),
                section_title=str(metadata.get("section_title", "Unknown")),
                source=str(metadata.get("source", "Unknown source")),
                distance=distance,
                relevance_score=self._distance_to_relevance_score(distance),
            )

            retrieved_chunks.append(retrieved_chunk)

        return retrieved_chunks


def print_retrieval_results(chunks: list[RetrievedChunk]):
    if not chunks:
        print("No results found.")
        return

    for chunk in chunks:
        print("=" * 100)
        print(f"Rank: {chunk.rank}")
        print(f"Source: {chunk.source}")
        print(f"Paper: {chunk.paper_title}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Distance: {chunk.distance:.4f}")
        print(f"Relevance Score: {chunk.relevance_score:.4f}")
        print("-" * 100)
        print(chunk.text[:1200])

        if len(chunk.text) > 1200:
            print("... [truncated]")

    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search PaperMind vector store with citation-ready retrieval."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User question or search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )

    parser.add_argument(
        "--paper-title",
        type=str,
        default=None,
        help="Optional exact paper title filter.",
    )

    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Optional exact file name filter.",
    )

    parser.add_argument(
        "--section-title",
        type=str,
        default=None,
        help="Optional exact section title filter.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print formatted LLM context block.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save retrieval results as JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    retriever = PaperRetriever()

    chunks, context = retriever.search_as_context(
        query=args.query,
        top_k=args.top_k,
        paper_title=args.paper_title,
        file_name=args.file_name,
        section_title=args.section_title,
    )

    print_retrieval_results(chunks)

    if args.show_context:
        print("\n\nLLM CONTEXT")
        print("=" * 100)
        print(context)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                [chunk.to_dict() for chunk in chunks],
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"\nSaved retrieval results to: {output_path}")


if __name__ == "__main__":
    main()