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
from src.config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger


logger = get_logger(__name__)


REVIEW_QUERIES = {
    "summary_and_contribution": (
        "What is the paper about, what problem does it solve, and what are its main contributions?"
    ),
    "method_and_architecture": (
        "What method, model architecture, algorithm, or framework does this paper propose?"
    ),
    "experiments_and_results": (
        "What datasets, tasks, baselines, metrics, experiments, and results are reported?"
    ),
    "limitations_and_weaknesses": (
        "What limitations, weaknesses, failure cases, negative results, or constraints are discussed?"
    ),
    "reproducibility": (
        "What implementation details, hyperparameters, code availability, dataset availability, "
        "or reproducibility information are provided?"
    ),
    "future_work": (
        "What future work, open problems, missing experiments, or next research directions are mentioned?"
    ),
}


@dataclass
class PaperReview:
    paper: str
    review_summary: str
    main_contributions: list[str]
    strengths: list[str]
    weaknesses: list[str]
    novelty_assessment: str
    technical_soundness: str
    experimental_quality: str
    missing_experiments: list[str]
    reproducibility_concerns: list[str]
    questions_for_authors: list[str]
    final_recommendation: str
    recommendation_reason: str
    reviewer_confidence: str
    sources: list[dict]
    llm_provider: str
    model_name: str
    raw_model_output: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ReviewerMode:
    """
    Generates a conference-style review for one selected research paper.

    This module retrieves evidence from the selected paper and asks the LLM to
    produce a structured review.

    It includes:
        1. JSON-only generation prompt
        2. JSON repair fallback
        3. Safe fallback review

    This is needed because local LLMs sometimes return prose instead of JSON.
    """

    def __init__(
        self,
        top_k_per_query: int = 2,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.2,
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
        file_name: Optional[str] = None,
    ) -> PaperReview:
        if file_name is None:
            indexed_files = self._get_indexed_file_names()

            if not indexed_files:
                raise RuntimeError("No indexed papers found in vector store.")

            if len(indexed_files) > 1:
                raise ValueError(
                    "Multiple papers are indexed. Please provide --file-name explicitly. "
                    f"Indexed files: {indexed_files}"
                )

            file_name = indexed_files[0]

        logger.info("Generating reviewer-mode report for: %s", file_name)

        chunks = self._retrieve_review_evidence(file_name=file_name)

        if not chunks:
            raise RuntimeError(
                f"No chunks were retrieved for {file_name}. "
                "Check the file name or rebuild the vector store."
            )

        context, sources = self._format_context_with_sources(chunks)

        prompt = self._build_review_prompt(
            file_name=file_name,
            context=context,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            parsed_output = self._parse_json_output(cleaned_output)

        except ValueError:
            logger.warning(
                "Reviewer output was not valid JSON. Attempting JSON repair."
            )

            repaired_output = self._repair_review_to_json(
                raw_output=cleaned_output,
                file_name=file_name,
            )

            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                parsed_output = self._parse_json_output(repaired_cleaned)
                raw_output = (
                    raw_output
                    + "\n\n--- JSON REPAIR OUTPUT ---\n\n"
                    + repaired_output
                )

            except ValueError:
                logger.warning(
                    "JSON repair failed. Using fallback reviewer output."
                )
                parsed_output = self._fallback_review(file_name=file_name)

        return PaperReview(
            paper=str(parsed_output.get("paper", file_name)),
            review_summary=str(
                parsed_output.get("review_summary", "Not found in retrieved context.")
            ),
            main_contributions=self._ensure_string_list(
                parsed_output.get("main_contributions", [])
            ),
            strengths=self._ensure_string_list(
                parsed_output.get("strengths", [])
            ),
            weaknesses=self._ensure_string_list(
                parsed_output.get("weaknesses", [])
            ),
            novelty_assessment=str(
                parsed_output.get("novelty_assessment", "Not found in retrieved context.")
            ),
            technical_soundness=str(
                parsed_output.get("technical_soundness", "Not found in retrieved context.")
            ),
            experimental_quality=str(
                parsed_output.get("experimental_quality", "Not found in retrieved context.")
            ),
            missing_experiments=self._ensure_string_list(
                parsed_output.get("missing_experiments", [])
            ),
            reproducibility_concerns=self._ensure_string_list(
                parsed_output.get("reproducibility_concerns", [])
            ),
            questions_for_authors=self._ensure_string_list(
                parsed_output.get("questions_for_authors", [])
            ),
            final_recommendation=str(
                parsed_output.get("final_recommendation", "Not found in retrieved context.")
            ),
            recommendation_reason=str(
                parsed_output.get("recommendation_reason", "Not found in retrieved context.")
            ),
            reviewer_confidence=str(
                parsed_output.get("reviewer_confidence", "Not found in retrieved context.")
            ),
            sources=sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output=raw_output,
        )

    def _retrieve_review_evidence(
        self,
        file_name: str,
    ) -> list[RetrievedChunk]:
        unique_chunks = {}

        for query_name, query in REVIEW_QUERIES.items():
            logger.info("  Review query dimension: %s", query_name)

            chunks = self.retriever.search(
                query=query,
                top_k=self.top_k_per_query,
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

    def _build_review_prompt(
        self,
        file_name: str,
        context: str,
    ) -> str:
        return f"""
You are PaperMind Reviewer Mode, acting like a careful AI/ML conference reviewer.

Your task is to generate a structured review for exactly one paper.

Selected file:
{file_name}

Rules:
1. Use only the retrieved context.
2. Do not invent details.
3. Every important claim should include source labels such as [Source 1], [Source 2].
4. If information is missing, write "Not found in retrieved context."
5. Return valid JSON only.
6. Do not use markdown.
7. Do not write text before or after the JSON.
8. Your response must start with {{ and end with }}.
9. Be critical but fair.
10. The final recommendation must be one of:
    - Strong Accept
    - Accept
    - Weak Accept
    - Borderline
    - Weak Reject
    - Reject
    - Not enough evidence in retrieved context

Required JSON schema:
{{
  "paper": "{file_name}",
  "review_summary": "...",
  "main_contributions": [
    "..."
  ],
  "strengths": [
    "..."
  ],
  "weaknesses": [
    "..."
  ],
  "novelty_assessment": "...",
  "technical_soundness": "...",
  "experimental_quality": "...",
  "missing_experiments": [
    "..."
  ],
  "reproducibility_concerns": [
    "..."
  ],
  "questions_for_authors": [
    "..."
  ],
  "final_recommendation": "...",
  "recommendation_reason": "...",
  "reviewer_confidence": "..."
}}

Retrieved Context:
{context}
""".strip()

    def _repair_review_to_json(
        self,
        raw_output: str,
        file_name: str,
    ) -> str:
        repair_prompt = f"""
Convert the following model output into valid JSON only.

Rules:
1. Your response must start with {{ and end with }}.
2. Do not use markdown.
3. Do not explain anything.
4. If a field is missing, use "Not found in retrieved context."

Selected file:
{file_name}

Required JSON schema:
{{
  "paper": "{file_name}",
  "review_summary": "...",
  "main_contributions": [
    "..."
  ],
  "strengths": [
    "..."
  ],
  "weaknesses": [
    "..."
  ],
  "novelty_assessment": "...",
  "technical_soundness": "...",
  "experimental_quality": "...",
  "missing_experiments": [
    "..."
  ],
  "reproducibility_concerns": [
    "..."
  ],
  "questions_for_authors": [
    "..."
  ],
  "final_recommendation": "...",
  "recommendation_reason": "...",
  "reviewer_confidence": "..."
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    @staticmethod
    def _fallback_review(file_name: str) -> dict:
        return {
            "paper": file_name,
            "review_summary": "Not reliably extracted.",
            "main_contributions": [
                "Not reliably extracted."
            ],
            "strengths": [
                "Not reliably extracted."
            ],
            "weaknesses": [
                "Not reliably extracted."
            ],
            "novelty_assessment": "Not reliably extracted.",
            "technical_soundness": "Not reliably extracted.",
            "experimental_quality": "Not reliably extracted.",
            "missing_experiments": [
                "Not reliably extracted."
            ],
            "reproducibility_concerns": [
                "Not reliably extracted."
            ],
            "questions_for_authors": [
                "Not reliably extracted."
            ],
            "final_recommendation": "Not enough evidence in retrieved context",
            "recommendation_reason": "The model output could not be reliably parsed.",
            "reviewer_confidence": "Low",
        }

    def _get_indexed_file_names(self) -> list[str]:
        vector_store = ChromaVectorStore(
            persist_dir=CHROMA_DB_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )

        collection_count = vector_store.count()

        if collection_count == 0:
            raise RuntimeError("Vector store is empty. Build the vector store first.")

        results = vector_store.collection.get(
            include=["metadatas"],
            limit=collection_count,
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

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _parse_json_output(text: str) -> dict:
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

    @staticmethod
    def _ensure_string_list(value) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(item) for item in value]

        return [str(value)]


def clean_markdown_cell(value) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("|", "/")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def review_to_markdown(review: PaperReview) -> str:
    lines = [
        "# Reviewer Mode Report",
        "",
        f"**Paper:** {review.paper}",
        f"**LLM Provider:** {review.llm_provider}",
        f"**Model:** {review.model_name}",
        "",
        "## Review Summary",
        review.review_summary,
        "",
        "## Main Contributions",
    ]

    for item in review.main_contributions:
        lines.append(f"- {item}")

    lines.extend(["", "## Strengths"])

    for item in review.strengths:
        lines.append(f"- {item}")

    lines.extend(["", "## Weaknesses"])

    for item in review.weaknesses:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Novelty Assessment",
            review.novelty_assessment,
            "",
            "## Technical Soundness",
            review.technical_soundness,
            "",
            "## Experimental Quality",
            review.experimental_quality,
            "",
            "## Missing Experiments",
        ]
    )

    for item in review.missing_experiments:
        lines.append(f"- {item}")

    lines.extend(["", "## Reproducibility Concerns"])

    for item in review.reproducibility_concerns:
        lines.append(f"- {item}")

    lines.extend(["", "## Questions for Authors"])

    for item in review.questions_for_authors:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Final Recommendation",
            review.final_recommendation,
            "",
            "## Recommendation Reason",
            review.recommendation_reason,
            "",
            "## Reviewer Confidence",
            review.reviewer_confidence,
            "",
            "## Sources",
        ]
    )

    for source in review.sources:
        lines.append(
            f"- **{source['source_id']}**: "
            f"{source['citation']} "
            f"(relevance={source['relevance_score']:.4f})"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a conference-style review for a research paper."
    )

    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Exact indexed PDF file name, e.g., evolvegc.pdf.",
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=2,
        help="Chunks retrieved per review query.",
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
        default=0.2,
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/reviewer_report.json",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/reviewer_report.md",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    reviewer = ReviewerMode(
        top_k_per_query=args.top_k_per_query,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
    )

    review = reviewer.generate(
        file_name=args.file_name,
    )

    review_dict = review.to_dict()
    markdown = review_to_markdown(review)

    print("\nREVIEWER MODE REPORT")
    print("=" * 100)
    print(markdown)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(review_dict, file, indent=4, ensure_ascii=False)

    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"\nSaved reviewer JSON to: {output_json_path}")
    print(f"Saved reviewer Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()