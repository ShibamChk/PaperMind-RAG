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


GAP_QUERIES = {
    "limitations": (
        "What limitations, weaknesses, failure cases, or constraints are discussed in this paper?"
    ),
    "future_work": (
        "What future work, open problems, or next research directions are mentioned?"
    ),
    "evaluation_gaps": (
        "What datasets, baselines, metrics, ablation studies, or evaluation settings are used or missing?"
    ),
    "scalability": (
        "Does the paper discuss scalability, efficiency, runtime, memory cost, or deployment challenges?"
    ),
    "robustness": (
        "Does the paper discuss robustness, generalization, noisy data, missing data, distribution shift, or real-world reliability?"
    ),
    "reproducibility": (
        "Does the paper provide code, hyperparameters, dataset details, implementation details, or reproducibility information?"
    ),
}


@dataclass
class PaperGapProfile:
    paper: str
    problem_area: str
    stated_limitations: str
    missing_or_weak_evaluation: str
    dataset_gaps: str
    scalability_gaps: str
    robustness_gaps: str
    reproducibility_gaps: str
    future_work: str
    possible_research_opportunities: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchGapReport:
    analyzed_files: list[str]
    paper_gap_profiles: list[dict]
    common_limitations: list[str]
    dataset_gaps: list[str]
    evaluation_gaps: list[str]
    scalability_gaps: list[str]
    robustness_gaps: list[str]
    reproducibility_gaps: list[str]
    proposed_research_directions: list[str]
    suggested_experiments: list[str]
    suggested_ablation_studies: list[str]
    overall_summary: str
    sources: list[dict]
    llm_provider: str
    model_name: str
    raw_model_output: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ResearchGapFinder:
    """
    Research Gap Finder for multiple papers.

    Design:
        1. Retrieve gap-related evidence separately for each paper.
        2. Generate a gap profile for each paper.
        3. Synthesize cross-paper research gaps and experiment ideas.

    This avoids the previous issue where the LLM focused on only one paper
    when all papers were mixed into one large prompt.
    """

    def __init__(
        self,
        top_k_per_query: int = 1,
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
        file_names: Optional[list[str]] = None,
    ) -> ResearchGapReport:
        if file_names is None or len(file_names) == 0:
            file_names = self._get_indexed_file_names()

        file_names = sorted(list(set(file_names)))

        if len(file_names) < 1:
            raise ValueError("At least one indexed paper is required.")

        paper_profiles = []
        all_sources = []
        raw_outputs = []

        source_counter = 1

        for file_name in file_names:
            logger.info("Generating research gap profile for paper: %s", file_name)

            chunks = self._retrieve_gap_evidence_for_file(file_name=file_name)

            if not chunks:
                logger.warning("No gap evidence found for %s", file_name)
                profile = self._fallback_profile(file_name=file_name)
                paper_profiles.append(profile.to_dict())
                continue

            context, sources, source_counter = self._format_context_with_sources(
                chunks=chunks,
                start_index=source_counter,
            )

            all_sources.extend(sources)

            profile, raw_output = self._generate_single_paper_gap_profile(
                file_name=file_name,
                context=context,
            )

            paper_profiles.append(profile.to_dict())
            raw_outputs.append(f"\n\n--- RAW GAP PROFILE FOR {file_name} ---\n{raw_output}")

        cross_report, cross_raw_output = self._generate_cross_paper_gap_report(
            paper_profiles=paper_profiles,
            file_names=file_names,
        )

        raw_outputs.append("\n\n--- RAW CROSS-PAPER GAP OUTPUT ---\n" + cross_raw_output)

        return ResearchGapReport(
            analyzed_files=file_names,
            paper_gap_profiles=paper_profiles,
            common_limitations=self._ensure_string_list(
                cross_report.get("common_limitations", [])
            ),
            dataset_gaps=self._ensure_string_list(
                cross_report.get("dataset_gaps", [])
            ),
            evaluation_gaps=self._ensure_string_list(
                cross_report.get("evaluation_gaps", [])
            ),
            scalability_gaps=self._ensure_string_list(
                cross_report.get("scalability_gaps", [])
            ),
            robustness_gaps=self._ensure_string_list(
                cross_report.get("robustness_gaps", [])
            ),
            reproducibility_gaps=self._ensure_string_list(
                cross_report.get("reproducibility_gaps", [])
            ),
            proposed_research_directions=self._ensure_string_list(
                cross_report.get("proposed_research_directions", [])
            ),
            suggested_experiments=self._ensure_string_list(
                cross_report.get("suggested_experiments", [])
            ),
            suggested_ablation_studies=self._ensure_string_list(
                cross_report.get("suggested_ablation_studies", [])
            ),
            overall_summary=str(
                cross_report.get(
                    "overall_summary",
                    "Not found in retrieved context.",
                )
            ),
            sources=all_sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output="\n".join(raw_outputs),
        )

    def _retrieve_gap_evidence_for_file(
        self,
        file_name: str,
    ) -> list[RetrievedChunk]:
        unique_chunks = {}

        for query_name, query in GAP_QUERIES.items():
            logger.info("  Gap query dimension: %s", query_name)

            chunks = self.retriever.search(
                query=query,
                top_k=self.top_k_per_query,
                file_name=file_name,
            )

            for chunk in chunks:
                if chunk.chunk_id not in unique_chunks:
                    unique_chunks[chunk.chunk_id] = chunk

        return list(unique_chunks.values())

    def _generate_single_paper_gap_profile(
        self,
        file_name: str,
        context: str,
    ) -> tuple[PaperGapProfile, str]:
        prompt = self._build_single_paper_gap_prompt(
            file_name=file_name,
            context=context,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            parsed_output = self._parse_json_output(cleaned_output)

        except ValueError:
            logger.warning(
                "Gap profile for %s was not valid JSON. Attempting repair.",
                file_name,
            )

            repaired_output = self._repair_gap_profile_to_json(
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
                    "JSON repair failed for %s. Using fallback profile.",
                    file_name,
                )
                return self._fallback_profile(file_name=file_name), raw_output

        profile = PaperGapProfile(
            paper=str(parsed_output.get("paper", file_name)),
            problem_area=str(
                parsed_output.get("problem_area", "Not found in retrieved context.")
            ),
            stated_limitations=str(
                parsed_output.get("stated_limitations", "Not found in retrieved context.")
            ),
            missing_or_weak_evaluation=str(
                parsed_output.get("missing_or_weak_evaluation", "Not found in retrieved context.")
            ),
            dataset_gaps=str(
                parsed_output.get("dataset_gaps", "Not found in retrieved context.")
            ),
            scalability_gaps=str(
                parsed_output.get("scalability_gaps", "Not found in retrieved context.")
            ),
            robustness_gaps=str(
                parsed_output.get("robustness_gaps", "Not found in retrieved context.")
            ),
            reproducibility_gaps=str(
                parsed_output.get("reproducibility_gaps", "Not found in retrieved context.")
            ),
            future_work=str(
                parsed_output.get("future_work", "Not found in retrieved context.")
            ),
            possible_research_opportunities=self._ensure_string_list(
                parsed_output.get("possible_research_opportunities", [])
            ),
        )

        return profile, raw_output

    def _generate_cross_paper_gap_report(
        self,
        paper_profiles: list[dict],
        file_names: list[str],
    ) -> tuple[dict, str]:
        prompt = self._build_cross_paper_gap_prompt(
            paper_profiles=paper_profiles,
            file_names=file_names,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            parsed_output = self._parse_json_output(cleaned_output)

        except ValueError:
            logger.warning(
                "Cross-paper gap report was not valid JSON. Attempting repair."
            )

            repaired_output = self._repair_cross_gap_report_to_json(
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
                logger.warning(
                    "Cross-paper JSON repair failed. Using fallback gap report."
                )

                parsed_output = self._fallback_cross_gap_report(
                    file_names=file_names,
                    paper_profiles=paper_profiles,
                )

        return parsed_output, raw_output

    def _build_single_paper_gap_prompt(
        self,
        file_name: str,
        context: str,
    ) -> str:
        return f"""
You are PaperMind, a research gap analysis assistant.

Your task is to extract a research-gap profile for exactly one paper.

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
  "problem_area": "...",
  "stated_limitations": "...",
  "missing_or_weak_evaluation": "...",
  "dataset_gaps": "...",
  "scalability_gaps": "...",
  "robustness_gaps": "...",
  "reproducibility_gaps": "...",
  "future_work": "...",
  "possible_research_opportunities": [
    "..."
  ]
}}

Retrieved Context:
{context}
""".strip()

    def _build_cross_paper_gap_prompt(
        self,
        paper_profiles: list[dict],
        file_names: list[str],
    ) -> str:
        selected_files = ", ".join(file_names)
        profiles_json = json.dumps(paper_profiles, indent=2, ensure_ascii=False)

        return f"""
You are PaperMind, a research supervisor.

Your task is to synthesize research gaps across multiple paper profiles.

Selected files:
{selected_files}

Rules:
1. Use only the structured paper profiles below.
2. Do not invent details.
3. Focus on actionable research opportunities.
4. Return valid JSON only.
5. Do not use markdown.
6. Do not write text before or after the JSON.
7. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "common_limitations": [
    "..."
  ],
  "dataset_gaps": [
    "..."
  ],
  "evaluation_gaps": [
    "..."
  ],
  "scalability_gaps": [
    "..."
  ],
  "robustness_gaps": [
    "..."
  ],
  "reproducibility_gaps": [
    "..."
  ],
  "proposed_research_directions": [
    "..."
  ],
  "suggested_experiments": [
    "..."
  ],
  "suggested_ablation_studies": [
    "..."
  ],
  "overall_summary": "..."
}}

Structured paper profiles:
{profiles_json}
""".strip()

    def _repair_gap_profile_to_json(
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
  "problem_area": "...",
  "stated_limitations": "...",
  "missing_or_weak_evaluation": "...",
  "dataset_gaps": "...",
  "scalability_gaps": "...",
  "robustness_gaps": "...",
  "reproducibility_gaps": "...",
  "future_work": "...",
  "possible_research_opportunities": [
    "..."
  ]
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    def _repair_cross_gap_report_to_json(
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
  "common_limitations": [
    "..."
  ],
  "dataset_gaps": [
    "..."
  ],
  "evaluation_gaps": [
    "..."
  ],
  "scalability_gaps": [
    "..."
  ],
  "robustness_gaps": [
    "..."
  ],
  "reproducibility_gaps": [
    "..."
  ],
  "proposed_research_directions": [
    "..."
  ],
  "suggested_experiments": [
    "..."
  ],
  "suggested_ablation_studies": [
    "..."
  ],
  "overall_summary": "..."
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
    def _fallback_profile(file_name: str) -> PaperGapProfile:
        return PaperGapProfile(
            paper=file_name,
            problem_area="Not reliably extracted.",
            stated_limitations="Not reliably extracted.",
            missing_or_weak_evaluation="Not reliably extracted.",
            dataset_gaps="Not reliably extracted.",
            scalability_gaps="Not reliably extracted.",
            robustness_gaps="Not reliably extracted.",
            reproducibility_gaps="Not reliably extracted.",
            future_work="Not reliably extracted.",
            possible_research_opportunities=[
                "Not reliably extracted."
            ],
        )

    @staticmethod
    def _fallback_cross_gap_report(
        file_names: list[str],
        paper_profiles: list[dict],
    ) -> dict:
        return {
            "common_limitations": [
                "Not reliably extracted."
            ],
            "dataset_gaps": [
                "Not reliably extracted."
            ],
            "evaluation_gaps": [
                "Not reliably extracted."
            ],
            "scalability_gaps": [
                "Not reliably extracted."
            ],
            "robustness_gaps": [
                "Not reliably extracted."
            ],
            "reproducibility_gaps": [
                "Not reliably extracted."
            ],
            "proposed_research_directions": [
                "Not reliably extracted."
            ],
            "suggested_experiments": [
                "Not reliably extracted."
            ],
            "suggested_ablation_studies": [
                "Not reliably extracted."
            ],
            "overall_summary": (
                "Structured cross-paper research gap synthesis could not be reliably generated. "
                "Use the paper gap profiles as the primary output."
            ),
        }

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


def gap_report_to_markdown(report: ResearchGapReport) -> str:
    lines = [
        "# Research Gap Report",
        "",
        f"**Analyzed files:** {', '.join(report.analyzed_files)}",
        f"**LLM Provider:** {report.llm_provider}",
        f"**Model:** {report.model_name}",
        "",
        "## Overall Summary",
        report.overall_summary,
        "",
        "## Paper-Level Gap Profiles",
        "",
        "| Paper | Problem Area | Stated Limitations | Weak Evaluation | Dataset Gaps | Scalability Gaps | Robustness Gaps | Reproducibility Gaps | Future Work |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for profile in report.paper_gap_profiles:
        lines.append(
            "| "
            + clean_markdown_cell(profile.get("paper", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("problem_area", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("stated_limitations", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("missing_or_weak_evaluation", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("dataset_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("scalability_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("robustness_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("reproducibility_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("future_work", "Not found"))
            + " |"
        )

    sections = [
        ("Common Limitations", report.common_limitations),
        ("Dataset Gaps", report.dataset_gaps),
        ("Evaluation Gaps", report.evaluation_gaps),
        ("Scalability Gaps", report.scalability_gaps),
        ("Robustness Gaps", report.robustness_gaps),
        ("Reproducibility Gaps", report.reproducibility_gaps),
        ("Proposed Research Directions", report.proposed_research_directions),
        ("Suggested Experiments", report.suggested_experiments),
        ("Suggested Ablation Studies", report.suggested_ablation_studies),
    ]

    for title, items in sections:
        lines.extend(["", f"## {title}"])

        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("- Not found in retrieved context.")

    lines.extend(["", "## Sources"])

    for source in report.sources:
        lines.append(
            f"- **{source['source_id']}**: "
            f"{source['citation']} "
            f"(relevance={source['relevance_score']:.4f})"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate research gap report from indexed research papers."
    )

    parser.add_argument(
        "--file-names",
        nargs="*",
        default=None,
        help=(
            "Optional list of exact file names to analyze. "
            "Example: --file-names evolvegc.pdf tgn.pdf"
        ),
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=1,
        help="Chunks retrieved per gap query per paper.",
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
        default="reports/research_gap_report.json",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/research_gap_report.md",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    finder = ResearchGapFinder(
        top_k_per_query=args.top_k_per_query,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
    )

    report = finder.generate(
        file_names=args.file_names,
    )

    report_dict = report.to_dict()
    markdown = gap_report_to_markdown(report)

    print("\nRESEARCH GAP REPORT")
    print("=" * 100)
    print(markdown)

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file, indent=4, ensure_ascii=False)

    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"\nSaved research gap JSON to: {output_json_path}")
    print(f"Saved research gap Markdown to: {output_md_path}")


if __name__ == "__main__":
    main()