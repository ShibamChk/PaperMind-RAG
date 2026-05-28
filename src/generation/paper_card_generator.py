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
from src.generation.answer_generator import AnswerGenerator
from src.utils.logger import get_logger


logger = get_logger(__name__)


PAPER_CARD_QUERIES = {
    "problem_motivation": (
        "What research problem, motivation, and research gap does this paper address?"
    ),
    "main_contribution": (
        "What are the main contributions and novelty of this paper?"
    ),
    "method_architecture": (
        "What proposed method, model architecture, algorithm, or framework does this paper introduce?"
    ),
    "datasets_tasks": (
        "What datasets, tasks, experimental setup, and evaluation settings are used in this paper?"
    ),
    "metrics_results": (
        "What evaluation metrics, baselines, and key experimental results are reported?"
    ),
    "limitations_future_work": (
        "What limitations, weaknesses, conclusions, and future work are discussed?"
    ),
    "reproducibility": (
        "What implementation details, hyperparameters, code availability, dataset availability, "
        "or reproducibility information are provided?"
    ),
}


@dataclass
class PaperCard:
    paper_title: str
    file_name: str
    problem: str
    motivation: str
    main_contribution: str
    proposed_method: str
    model_architecture: str
    datasets: str
    tasks: str
    evaluation_metrics: str
    baselines: str
    key_results: str
    limitations: str
    future_work: str
    reproducibility_notes: str
    sources: list[dict]
    llm_provider: str
    model_name: str
    raw_model_output: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class PaperCardGenerator:
    """
    Generates a structured Paper Card for one research paper.

    This is different from generic RAG Q&A:
    it retrieves targeted evidence for multiple research dimensions and then
    asks the LLM to produce a structured research profile.
    """

    def __init__(
        self,
        top_k_per_query: int = 3,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.top_k_per_query = top_k_per_query

        self.retriever = PaperRetriever()

        self.answer_generator = AnswerGenerator(
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=temperature,
        )

    def generate(
        self,
        paper_title: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> PaperCard:
        retrieved_chunks = self._retrieve_paper_evidence(
            paper_title=paper_title,
            file_name=file_name,
        )

        if not retrieved_chunks:
            raise RuntimeError(
                "No chunks were retrieved for the paper. "
                "Check paper_title/file_name filters or rebuild the vector store."
            )

        context, sources = self._format_context_with_sources(retrieved_chunks)

        inferred_paper_title = sources[0].get("paper_title", paper_title or "Unknown")
        inferred_file_name = sources[0].get("file_name", file_name or "Unknown")

        prompt = self._build_paper_card_prompt(
            context=context,
            paper_title=inferred_paper_title,
            file_name=inferred_file_name,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        cleaned_output = self._strip_thinking_tags(raw_output)

        parsed_card = self._parse_json_output(cleaned_output)

        return PaperCard(
            paper_title=str(parsed_card.get("paper_title", inferred_paper_title)),
            file_name=str(parsed_card.get("file_name", inferred_file_name)),
            problem=str(parsed_card.get("problem", "Not found in retrieved context.")),
            motivation=str(parsed_card.get("motivation", "Not found in retrieved context.")),
            main_contribution=str(parsed_card.get("main_contribution", "Not found in retrieved context.")),
            proposed_method=str(parsed_card.get("proposed_method", "Not found in retrieved context.")),
            model_architecture=str(parsed_card.get("model_architecture", "Not found in retrieved context.")),
            datasets=str(parsed_card.get("datasets", "Not found in retrieved context.")),
            tasks=str(parsed_card.get("tasks", "Not found in retrieved context.")),
            evaluation_metrics=str(parsed_card.get("evaluation_metrics", "Not found in retrieved context.")),
            baselines=str(parsed_card.get("baselines", "Not found in retrieved context.")),
            key_results=str(parsed_card.get("key_results", "Not found in retrieved context.")),
            limitations=str(parsed_card.get("limitations", "Not found in retrieved context.")),
            future_work=str(parsed_card.get("future_work", "Not found in retrieved context.")),
            reproducibility_notes=str(parsed_card.get("reproducibility_notes", "Not found in retrieved context.")),
            sources=sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output=raw_output,
        )

    def _retrieve_paper_evidence(
        self,
        paper_title: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        unique_chunks = {}

        for query_name, query in PAPER_CARD_QUERIES.items():
            logger.info("Retrieving evidence for: %s", query_name)

            chunks = self.retriever.search(
                query=query,
                top_k=self.top_k_per_query,
                paper_title=paper_title,
                file_name=file_name,
            )

            for chunk in chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk

        return list(unique_chunks.values())

    def _format_context_with_sources(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[dict]]:
        context_blocks = []
        sources = []

        for source_index, chunk in enumerate(chunks, start=1):
            source_id = f"Source {source_index}"

            block = (
                f"[{source_id}]\n"
                f"Paper: {chunk.paper_title}\n"
                f"File: {chunk.file_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Section: {chunk.section_title}\n"
                f"Citation: {chunk.source}\n"
                f"Relevance Score: {chunk.relevance_score:.4f}\n"
                f"Text:\n{chunk.text}"
            )

            context_blocks.append(block)

            sources.append(
                {
                    "source_id": source_id,
                    "paper_title": chunk.paper_title,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "citation": chunk.source,
                    "relevance_score": chunk.relevance_score,
                    "chunk_id": chunk.chunk_id,
                }
            )

        return "\n\n---\n\n".join(context_blocks), sources

    def _build_paper_card_prompt(
        self,
        context: str,
        paper_title: str,
        file_name: str,
    ) -> str:
        return f"""
You are PaperMind, a research-paper analysis assistant.

Your task is to generate a structured Paper Card for the given research paper.

Use only the retrieved context.
Do not invent details.
Every field should include source labels such as [Source 1], [Source 2] when evidence is available.
If a field is not supported by the retrieved context, write: "Not found in retrieved context."

Return valid JSON only.
Do not use markdown.
Do not add explanations outside JSON.

Paper title hint: {paper_title}
File name hint: {file_name}

Required JSON schema:
{{
  "paper_title": "...",
  "file_name": "...",
  "problem": "...",
  "motivation": "...",
  "main_contribution": "...",
  "proposed_method": "...",
  "model_architecture": "...",
  "datasets": "...",
  "tasks": "...",
  "evaluation_metrics": "...",
  "baselines": "...",
  "key_results": "...",
  "limitations": "...",
  "future_work": "...",
  "reproducibility_notes": "..."
}}

Retrieved Context:
{context}
""".strip()

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """
        Some local reasoning models may output <think>...</think>.
        Remove it before JSON parsing.
        """
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _parse_json_output(text: str) -> dict:
        """
        Parse JSON from LLM output.

        Handles:
        - raw JSON
        - ```json fenced JSON
        - extra text before/after JSON
        """
        text = text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if first_brace == -1 or last_brace == -1:
            raise ValueError(
                "Model output did not contain a JSON object.\n"
                f"Output:\n{text}"
            )

        json_text = text[first_brace:last_brace + 1]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Failed to parse model output as JSON.\n"
                f"JSON text:\n{json_text}"
            ) from error


def paper_card_to_markdown(card: PaperCard) -> str:
    """
    Convert a PaperCard object into a clean Markdown report.
    """
    lines = [
        f"# Paper Card: {card.paper_title}",
        "",
        f"**File:** {card.file_name}",
        f"**LLM Provider:** {card.llm_provider}",
        f"**Model:** {card.model_name}",
        "",
        "## Problem",
        card.problem,
        "",
        "## Motivation",
        card.motivation,
        "",
        "## Main Contribution",
        card.main_contribution,
        "",
        "## Proposed Method",
        card.proposed_method,
        "",
        "## Model Architecture",
        card.model_architecture,
        "",
        "## Datasets",
        card.datasets,
        "",
        "## Tasks",
        card.tasks,
        "",
        "## Evaluation Metrics",
        card.evaluation_metrics,
        "",
        "## Baselines",
        card.baselines,
        "",
        "## Key Results",
        card.key_results,
        "",
        "## Limitations",
        card.limitations,
        "",
        "## Future Work",
        card.future_work,
        "",
        "## Reproducibility Notes",
        card.reproducibility_notes,
        "",
        "## Sources",
    ]

    for source in card.sources:
        lines.append(
            f"- **{source['source_id']}**: "
            f"{source['citation']} "
            f"(relevance={source['relevance_score']:.4f})"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a structured Paper Card from indexed research papers."
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
        help="Optional exact file name filter, e.g., evolvegc.pdf.",
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=3,
        help="Chunks retrieved per Paper Card query.",
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["ollama", "openai"],
    )

    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/paper_card.json",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/paper_card.md",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generator = PaperCardGenerator(
        top_k_per_query=args.top_k_per_query,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
    )

    card = generator.generate(
        paper_title=args.paper_title,
        file_name=args.file_name,
    )

    card_dict = card.to_dict()
    markdown = paper_card_to_markdown(card)

    print("\nPAPER CARD")
    print("=" * 100)
    print(markdown)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(card_dict, file, indent=4, ensure_ascii=False)

    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"\nSaved Paper Card JSON to: {output_json_path}")
    print(f"Saved Paper Card Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()