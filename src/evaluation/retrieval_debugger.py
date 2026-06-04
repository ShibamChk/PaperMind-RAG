from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retriever import PaperRetriever, RetrievedChunk
from src.retrieval.vector_store import ChromaVectorStore
from src.config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.utils.logger import get_logger


logger = get_logger(__name__)


PAPER_CARD_DEBUG_QUERIES = {
    "problem": "What problem does this paper solve?",
    "motivation": (
        "What motivates this work? What challenge, need, or weakness in existing work "
        "does the paper describe?"
    ),
    "research_gap": (
        "What research gap does this paper address? What is missing, limited, or weak "
        "in previous methods?"
    ),
    "main_contribution": "What are the main contributions of this paper?",
    "method": "What method, model, algorithm, framework, or architecture does this paper propose?",
    "datasets": "What datasets, benchmarks, or data sources are used in this paper?",
    "metrics": "What evaluation metrics are used in this paper?",
    "baselines": "What baseline methods or comparison models are used?",
    "results": "What are the main experimental results or findings?",
    "limitations": "What limitations, weaknesses, or failure cases are discussed?",
    "future_work": "What future work or next research directions are discussed?",
    "reproducibility": (
        "What implementation details, hyperparameters, code availability, or dataset "
        "availability information are provided?"
    ),
}


GAP_DEBUG_QUERIES = {
    "limitations": "What limitations, weaknesses, failure cases, or constraints are discussed?",
    "evaluation_gaps": "What evaluation weaknesses, missing baselines, missing datasets, or missing ablations exist?",
    "dataset_gaps": "What dataset limitations, data constraints, or benchmark limitations are discussed?",
    "scalability": "Does the paper discuss scalability, runtime, memory, efficiency, or deployment challenges?",
    "robustness": "Does the paper discuss robustness, noisy data, missing data, or distribution shift?",
    "future_work": "What future work or open problems are mentioned?",
}


REVIEW_DEBUG_QUERIES = {
    "summary": "What is this paper about and what is its main contribution?",
    "strengths": "What are the strengths of this paper?",
    "weaknesses": "What are the weaknesses or limitations of this paper?",
    "experiments": "What experiments, datasets, baselines, and metrics are used?",
    "missing_experiments": "What experiments or ablation studies are missing or could improve the paper?",
    "reproducibility": "What reproducibility details are provided?",
}


PRESET_QUERY_GROUPS = {
    "paper_card": PAPER_CARD_DEBUG_QUERIES,
    "gap_finder": GAP_DEBUG_QUERIES,
    "reviewer": REVIEW_DEBUG_QUERIES,
}


@dataclass
class DebugResult:
    query_name: str
    query: str
    rank: int
    chunk_id: str
    file_name: str
    paper_title: str
    page_number: int
    section_title: str
    source: str
    relevance_score: float
    distance: float
    text_preview: str
    full_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_preview(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def get_indexed_files() -> list[str]:
    vector_store = ChromaVectorStore(
        persist_dir=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    count = vector_store.count()

    if count == 0:
        return []

    results = vector_store.collection.get(
        include=["metadatas"],
        limit=count,
    )

    metadatas = results.get("metadatas", [])

    file_names = sorted(
        {
            metadata.get("file_name")
            for metadata in metadatas
            if metadata.get("file_name")
        }
    )

    return file_names


def get_section_distribution(file_name: Optional[str] = None) -> dict[str, int]:
    vector_store = ChromaVectorStore(
        persist_dir=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    count = vector_store.count()

    if count == 0:
        return {}

    results = vector_store.collection.get(
        include=["metadatas"],
        limit=count,
    )

    metadatas = results.get("metadatas", [])

    section_counts: dict[str, int] = {}

    for metadata in metadatas:
        if file_name and metadata.get("file_name") != file_name:
            continue

        section = metadata.get("section_title", "Unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    return dict(
        sorted(
            section_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def run_query_debug(
    retriever: PaperRetriever,
    query_name: str,
    query: str,
    top_k: int,
    file_name: Optional[str] = None,
    section_title: Optional[str] = None,
) -> list[DebugResult]:
    chunks = retriever.search(
        query=query,
        top_k=top_k,
        file_name=file_name,
        section_title=section_title,
    )

    debug_results = []

    for chunk in chunks:
        debug_results.append(
            DebugResult(
                query_name=query_name,
                query=query,
                rank=chunk.rank,
                chunk_id=chunk.chunk_id,
                file_name=chunk.file_name,
                paper_title=chunk.paper_title,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                source=chunk.source,
                relevance_score=chunk.relevance_score,
                distance=chunk.distance,
                text_preview=clean_preview(chunk.text),
                full_text=chunk.text,
            )
        )

    return debug_results


def print_debug_results(results: list[DebugResult], show_full_text: bool = False):
    if not results:
        print("No retrieval results found.")
        return

    current_query_name = None

    for result in results:
        if result.query_name != current_query_name:
            current_query_name = result.query_name
            print("\n")
            print("=" * 120)
            print(f"QUERY GROUP: {result.query_name}")
            print(f"QUERY: {result.query}")
            print("=" * 120)

        print("-" * 120)
        print(f"Rank: {result.rank}")
        print(f"File: {result.file_name}")
        print(f"Paper: {result.paper_title}")
        print(f"Page: {result.page_number}")
        print(f"Section: {result.section_title}")
        print(f"Source: {result.source}")
        print(f"Relevance Score: {result.relevance_score:.4f}")
        print(f"Distance: {result.distance:.4f}")
        print("-" * 120)

        if show_full_text:
            print(result.full_text)
        else:
            print(result.text_preview)

    print("-" * 120)


def results_to_markdown(results: list[DebugResult], show_full_text: bool = False) -> str:
    lines = [
        "# Retrieval Debug Report",
        "",
        "This report shows the exact chunks retrieved from the vector store for each query.",
        "",
    ]

    current_query_name = None

    for result in results:
        if result.query_name != current_query_name:
            current_query_name = result.query_name
            lines.extend(
                [
                    "",
                    f"## Query Group: {result.query_name}",
                    "",
                    f"**Query:** {result.query}",
                    "",
                ]
            )

        lines.extend(
            [
                f"### Rank {result.rank}",
                "",
                f"- **File:** `{result.file_name}`",
                f"- **Paper:** {result.paper_title}",
                f"- **Page:** {result.page_number}",
                f"- **Section:** {result.section_title}",
                f"- **Source:** {result.source}",
                f"- **Relevance Score:** {result.relevance_score:.4f}",
                f"- **Distance:** {result.distance:.4f}",
                "",
                "**Retrieved Text:**",
                "",
                "```text",
                result.full_text if show_full_text else result.text_preview,
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def save_outputs(
    results: list[DebugResult],
    output_json: Optional[str] = None,
    output_md: Optional[str] = None,
    show_full_text: bool = False,
):
    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(
                [result.to_dict() for result in results],
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(f"\nSaved JSON debug report to: {output_json_path}")

    if output_md:
        output_md_path = Path(output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        markdown = results_to_markdown(
            results=results,
            show_full_text=show_full_text,
        )

        with open(output_md_path, "w", encoding="utf-8") as file:
            file.write(markdown)

        print(f"Saved Markdown debug report to: {output_md_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug retrieval results for PaperMind-RAG."
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single custom query to debug.",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=list(PRESET_QUERY_GROUPS.keys()),
        help="Run a group of predefined debug queries.",
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
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for debugging. This does not call the LLM.",
    )

    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List indexed files and exit.",
    )

    parser.add_argument(
        "--section-report",
        action="store_true",
        help="Print section distribution from Chroma metadata.",
    )

    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Print/save full chunk text instead of preview.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save debug results as JSON.",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional path to save debug results as Markdown.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_files:
        indexed_files = get_indexed_files()

        print("\nIndexed files:")
        for file_name in indexed_files:
            print(f"- {file_name}")

        return

    if args.section_report:
        section_counts = get_section_distribution(file_name=args.file_name)

        print("\nSection distribution:")
        for section, count in section_counts.items():
            print(f"{section}: {count}")

        return

    if not args.query and not args.preset:
        raise ValueError("Provide either --query or --preset.")

    retriever = PaperRetriever()

    all_results: list[DebugResult] = []

    if args.query:
        all_results.extend(
            run_query_debug(
                retriever=retriever,
                query_name="custom_query",
                query=args.query,
                top_k=args.top_k,
                file_name=args.file_name,
                section_title=args.section_title,
            )
        )

    if args.preset:
        query_group = PRESET_QUERY_GROUPS[args.preset]

        for query_name, query in query_group.items():
            all_results.extend(
                run_query_debug(
                    retriever=retriever,
                    query_name=query_name,
                    query=query,
                    top_k=args.top_k,
                    file_name=args.file_name,
                    section_title=args.section_title,
                )
            )

    print_debug_results(
        results=all_results,
        show_full_text=args.show_full_text,
    )

    save_outputs(
        results=all_results,
        output_json=args.output_json,
        output_md=args.output_md,
        show_full_text=args.show_full_text,
    )


if __name__ == "__main__":
    main()