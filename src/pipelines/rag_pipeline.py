from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.retriever import PaperRetriever
from src.generation.answer_generator import AnswerGenerator


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Steps:
        1. Retrieve citation-ready chunks
        2. Generate answer using only retrieved context
        3. Return answer with sources
    """

    def __init__(
        self,
        top_k: int = 5,
        llm_model_name: str | None = None,
        llm_provider: str | None = None,
        temperature: float = 0.2,
    ):
        self.top_k = top_k

        self.retriever = PaperRetriever()

        self.answer_generator = AnswerGenerator(
            model_name=llm_model_name,
            provider=llm_provider,
            temperature=temperature,
        )

    def ask(
        self,
        question: str,
        paper_title: str | None = None,
        file_name: str | None = None,
        section_title: str | None = None,
    ) -> dict:
        retrieved_chunks = self.retriever.search(
            query=question,
            top_k=self.top_k,
            paper_title=paper_title,
            file_name=file_name,
            section_title=section_title,
        )

        rag_answer = self.answer_generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return rag_answer.to_dict()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask a citation-grounded question over indexed research papers."
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask over the indexed papers.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks.",
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["ollama", "openai"],
        help="LLM provider. If omitted, uses .env LLM_PROVIDER.",
    )

    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=None,
        help="Optional model name. Example: qwen3:14b or gpt-4o-mini.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--paper-title",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--section-title",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save answer JSON.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = RAGPipeline(
        top_k=args.top_k,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
    )

    result = pipeline.ask(
        question=args.question,
        paper_title=args.paper_title,
        file_name=args.file_name,
        section_title=args.section_title,
    )

    print("\nQUESTION")
    print("=" * 100)
    print(result["question"])

    print("\nANSWER")
    print("=" * 100)
    print(result["answer"])

    print("\nLLM")
    print("=" * 100)
    print(f"Provider: {result['llm_provider']}")
    print(f"Model: {result['model_name']}")

    print("\nSOURCES")
    print("=" * 100)
    print(json.dumps(result["sources"], indent=4, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

        print(f"\nSaved RAG answer to: {output_path}")


if __name__ == "__main__":
    main()