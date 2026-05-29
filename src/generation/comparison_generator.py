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


COMPARISON_QUERIES = {
    "problem_and_contribution": (
        "What problem does this paper solve and what are its main contributions?"
    ),
    "method_and_architecture": (
        "What method, model architecture, algorithm, or framework does this paper propose?"
    ),
    "datasets_tasks_metrics": (
        "What datasets, tasks, evaluation metrics, baselines, and experimental setup are used?"
    ),
    "results_and_findings": (
        "What are the key results, findings, and performance comparisons reported?"
    ),
    "limitations_and_future_work": (
        "What limitations, weaknesses, conclusions, or future work are discussed?"
    ),
}


@dataclass
class PaperComparison:
    compared_files: list[str]
    comparison_table: list[dict]
    cross_paper_summary: str
    key_similarities: list[str]
    key_differences: list[str]
    research_gaps: list[str]
    sources: list[dict]
    llm_provider: str
    model_name: str
    raw_model_output: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MultiPaperComparisonGenerator:
    """
    Generates a multi-paper comparison matrix.

    Improved design:
        1. Retrieve evidence separately for each paper.
        2. Generate one structured row per paper.
        3. Compare the generated rows.

    This prevents the LLM from ignoring one paper when multiple papers are mixed
    into one large prompt.
    """

    def __init__(
        self,
        top_k_per_query: int = 2,
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
        file_names: Optional[list[str]] = None,
    ) -> PaperComparison:
        if file_names is None or len(file_names) == 0:
            file_names = self._get_indexed_file_names()

        file_names = sorted(list(set(file_names)))

        if len(file_names) < 2:
            logger.warning(
                "Only %s paper(s) selected. Multi-paper comparison is more useful with at least 2 papers.",
                len(file_names),
            )

        comparison_rows = []
        all_sources = []
        raw_outputs = []

        source_counter = 1

        for file_name in file_names:
            logger.info("Generating comparison row for paper: %s", file_name)

            chunks = self._retrieve_evidence_for_file(file_name=file_name)

            if not chunks:
                logger.warning("No chunks found for %s", file_name)
                comparison_rows.append(self._fallback_row(file_name=file_name))
                continue

            context, sources, source_counter = self._format_context_with_sources(
                chunks=chunks,
                start_index=source_counter,
            )

            all_sources.extend(sources)

            row, raw_output = self._generate_single_paper_row(
                file_name=file_name,
                context=context,
            )

            comparison_rows.append(row)
            raw_outputs.append(f"\n\n--- RAW OUTPUT FOR {file_name} ---\n{raw_output}")

        cross_analysis, cross_raw_output = self._generate_cross_paper_analysis(
            comparison_rows=comparison_rows,
            file_names=file_names,
        )

        raw_outputs.append("\n\n--- RAW CROSS-PAPER OUTPUT ---\n" + cross_raw_output)

        return PaperComparison(
            compared_files=file_names,
            comparison_table=comparison_rows,
            cross_paper_summary=cross_analysis.get(
                "cross_paper_summary",
                "Not found in retrieved context.",
            ),
            key_similarities=self._ensure_string_list(
                cross_analysis.get("key_similarities", [])
            ),
            key_differences=self._ensure_string_list(
                cross_analysis.get("key_differences", [])
            ),
            research_gaps=self._ensure_string_list(
                cross_analysis.get("research_gaps", [])
            ),
            sources=all_sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output="\n".join(raw_outputs),
        )

    def _retrieve_evidence_for_file(
        self,
        file_name: str,
    ) -> list[RetrievedChunk]:
        unique_chunks = {}

        for query_name, query in COMPARISON_QUERIES.items():
            logger.info("  Query dimension: %s", query_name)

            chunks = self.retriever.search(
                query=query,
                top_k=self.top_k_per_query,
                file_name=file_name,
            )

            for chunk in chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk

        return list(unique_chunks.values())

    def _generate_single_paper_row(
        self,
        file_name: str,
        context: str,
    ) -> tuple[dict, str]:
        prompt = self._build_single_paper_row_prompt(
            file_name=file_name,
            context=context,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            parsed_row = self._parse_json_output(cleaned_output)

        except ValueError:
            logger.warning(
                "Single-paper row for %s was not valid JSON. Attempting repair.",
                file_name,
            )

            repaired_output = self._repair_row_output_to_json(
                raw_output=cleaned_output,
                file_name=file_name,
            )

            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                parsed_row = self._parse_json_output(repaired_cleaned)
                raw_output = (
                    raw_output
                    + "\n\n--- JSON REPAIR OUTPUT ---\n\n"
                    + repaired_output
                )

            except ValueError:
                logger.warning(
                    "JSON repair failed for %s. Using fallback row.",
                    file_name,
                )
                parsed_row = self._fallback_row(file_name=file_name)

        return self._normalize_row(parsed_row, file_name=file_name), raw_output

    def _generate_cross_paper_analysis(
        self,
        comparison_rows: list[dict],
        file_names: list[str],
    ) -> tuple[dict, str]:
        prompt = self._build_cross_paper_prompt(
            comparison_rows=comparison_rows,
            file_names=file_names,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            parsed_output = self._parse_json_output(cleaned_output)

        except ValueError:
            logger.warning("Cross-paper output was not valid JSON. Attempting repair.")

            repaired_output = self._repair_cross_output_to_json(
                raw_output=cleaned_output,
                file_names=file_names,
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
                logger.warning("Cross-paper JSON repair failed. Using fallback.")
                parsed_output = self._fallback_cross_analysis(
                    comparison_rows=comparison_rows,
                    file_names=file_names,
                )

        return parsed_output, raw_output

    def _build_single_paper_row_prompt(
        self,
        file_name: str,
        context: str,
    ) -> str:
        return f"""
You are PaperMind, a research-paper analysis assistant.

Your task is to extract one structured comparison row for exactly one paper.

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

Required JSON schema:
{{
  "paper": "{file_name}",
  "problem": "...",
  "main_contribution": "...",
  "method": "...",
  "architecture_or_framework": "...",
  "datasets": "...",
  "tasks": "...",
  "evaluation_metrics": "...",
  "baselines": "...",
  "key_results": "...",
  "limitations": "...",
  "unique_strength": "..."
}}

Retrieved Context:
{context}
""".strip()

    def _build_cross_paper_prompt(
        self,
        comparison_rows: list[dict],
        file_names: list[str],
    ) -> str:
        selected_files = ", ".join(file_names)
        rows_json = json.dumps(comparison_rows, indent=2, ensure_ascii=False)

        return f"""
You are PaperMind, a research-paper comparison assistant.

Your task is to compare the structured rows for multiple papers.

Selected files:
{selected_files}

Rules:
1. Use only the structured rows provided below.
2. Do not invent details.
3. Return valid JSON only.
4. Do not use markdown.
5. Do not write text before or after the JSON.
6. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "cross_paper_summary": "...",
  "key_similarities": [
    "..."
  ],
  "key_differences": [
    "..."
  ],
  "research_gaps": [
    "..."
  ]
}}

Structured comparison rows:
{rows_json}
""".strip()

    def _repair_row_output_to_json(
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
  "problem": "...",
  "main_contribution": "...",
  "method": "...",
  "architecture_or_framework": "...",
  "datasets": "...",
  "tasks": "...",
  "evaluation_metrics": "...",
  "baselines": "...",
  "key_results": "...",
  "limitations": "...",
  "unique_strength": "..."
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    def _repair_cross_output_to_json(
        self,
        raw_output: str,
        file_names: list[str],
    ) -> str:
        selected_files = ", ".join(file_names)

        repair_prompt = f"""
Convert the following model output into valid JSON only.

Rules:
1. Your response must start with {{ and end with }}.
2. Do not use markdown.
3. Do not explain anything.
4. If a field is missing, use "Not found in retrieved context."

Selected files:
{selected_files}

Required JSON schema:
{{
  "cross_paper_summary": "...",
  "key_similarities": [
    "..."
  ],
  "key_differences": [
    "..."
  ],
  "research_gaps": [
    "..."
  ]
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    def _format_context_with_sources(
        self,
        chunks: list[RetrievedChunk],
        start_index: int,
    ) -> tuple[str, list[dict], int]:
        context_blocks = []
        sources = []

        source_index = start_index

        for chunk in chunks:
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

            source_index += 1

        return "\n\n---\n\n".join(context_blocks), sources, source_index

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
    def _fallback_row(file_name: str) -> dict:
        return {
            "paper": file_name,
            "problem": "Not reliably extracted.",
            "main_contribution": "Not reliably extracted.",
            "method": "Not reliably extracted.",
            "architecture_or_framework": "Not reliably extracted.",
            "datasets": "Not reliably extracted.",
            "tasks": "Not reliably extracted.",
            "evaluation_metrics": "Not reliably extracted.",
            "baselines": "Not reliably extracted.",
            "key_results": "Not reliably extracted.",
            "limitations": "Not reliably extracted.",
            "unique_strength": "Not reliably extracted.",
        }

    @staticmethod
    def _fallback_cross_analysis(
        comparison_rows: list[dict],
        file_names: list[str],
    ) -> dict:
        return {
            "cross_paper_summary": (
                "Structured cross-paper analysis could not be reliably generated. "
                "Use the comparison table rows as the primary output."
            ),
            "key_similarities": [
                "Not reliably extracted."
            ],
            "key_differences": [
                "Not reliably extracted."
            ],
            "research_gaps": [
                "Not reliably extracted."
            ],
        }

    @staticmethod
    def _normalize_row(row: dict, file_name: str) -> dict:
        required_keys = [
            "paper",
            "problem",
            "main_contribution",
            "method",
            "architecture_or_framework",
            "datasets",
            "tasks",
            "evaluation_metrics",
            "baselines",
            "key_results",
            "limitations",
            "unique_strength",
        ]

        normalized = {}

        for key in required_keys:
            value = row.get(key, "Not found in retrieved context.")

            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)

            normalized[key] = str(value)

        if not normalized["paper"] or normalized["paper"] == "Not found in retrieved context.":
            normalized["paper"] = file_name

        return normalized

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


def comparison_to_markdown(comparison: PaperComparison) -> str:
    lines = [
        "# Multi-Paper Comparison Matrix",
        "",
        f"**Compared files:** {', '.join(comparison.compared_files)}",
        f"**LLM Provider:** {comparison.llm_provider}",
        f"**Model:** {comparison.model_name}",
        "",
        "## Comparison Table",
        "",
        "| Paper | Problem | Main Contribution | Method | Architecture / Framework | Datasets | Tasks | Metrics | Baselines | Key Results | Limitations | Unique Strength |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for row in comparison.comparison_table:
        lines.append(
            "| "
            + clean_markdown_cell(row.get("paper", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("problem", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("main_contribution", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("method", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("architecture_or_framework", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("datasets", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("tasks", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("evaluation_metrics", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("baselines", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("key_results", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("limitations", "Not found"))
            + " | "
            + clean_markdown_cell(row.get("unique_strength", "Not found"))
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cross-Paper Summary",
            comparison.cross_paper_summary,
            "",
            "## Key Similarities",
        ]
    )

    if comparison.key_similarities:
        for item in comparison.key_similarities:
            lines.append(f"- {item}")
    else:
        lines.append("- Not found in retrieved context.")

    lines.extend(["", "## Key Differences"])

    if comparison.key_differences:
        for item in comparison.key_differences:
            lines.append(f"- {item}")
    else:
        lines.append("- Not found in retrieved context.")

    lines.extend(["", "## Research Gaps"])

    if comparison.research_gaps:
        for item in comparison.research_gaps:
            lines.append(f"- {item}")
    else:
        lines.append("- Not found in retrieved context.")

    lines.extend(["", "## Sources"])

    for source in comparison.sources:
        lines.append(
            f"- **{source['source_id']}**: "
            f"{source['citation']} "
            f"(relevance={source['relevance_score']:.4f})"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a multi-paper comparison matrix."
    )

    parser.add_argument(
        "--file-names",
        nargs="*",
        default=None,
        help=(
            "Optional list of exact file names to compare. "
            "Example: --file-names evolvegc.pdf dysat.pdf"
        ),
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=2,
        help="Chunks retrieved per comparison query per paper.",
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
        default="reports/multi_paper_comparison.json",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/multi_paper_comparison.md",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generator = MultiPaperComparisonGenerator(
        top_k_per_query=args.top_k_per_query,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
    )

    comparison = generator.generate(
        file_names=args.file_names,
    )

    comparison_dict = comparison.to_dict()
    markdown = comparison_to_markdown(comparison)

    print("\nMULTI-PAPER COMPARISON")
    print("=" * 100)
    print(markdown)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(comparison_dict, file, indent=4, ensure_ascii=False)

    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"\nSaved comparison JSON to: {output_json_path}")
    print(f"Saved comparison Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()