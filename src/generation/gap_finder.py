from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.generation.answer_generator import AnswerGenerator
from src.retrieval.evidence_builder import EvidenceBuilder, EvidencePack
from src.retrieval.vector_store import ChromaVectorStore
from src.config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.utils.logger import get_logger


logger = get_logger(__name__)


GAP_EVIDENCE_FIELDS = [
    "research_gap",
    "limitations",
    "future_work",
    "datasets",
    "metrics",
    "results",
    "reproducibility",
    "method",
]


@dataclass
class PaperGapProfile:
    paper: str
    problem_area: str
    stated_or_implied_research_gap: str
    dataset_gaps: str
    evaluation_gaps: str
    scalability_gaps: str
    robustness_gaps: str
    reproducibility_gaps: str
    limitations: str
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
    Research Gap Finder v2.

    v1 used direct vector retrieval for gap-related queries.
    v2 uses EvidenceBuilder to create section-aware, keyword-aware,
    field-specific evidence packs before generating research gaps.

    This is more robust for papers where:
        - motivation is inside the introduction
        - gaps are implicit in related work
        - future work is inside the conclusion
        - limitations are indirectly implied by experiments or future work
    """

    def __init__(
        self,
        top_k_per_query: int = 1,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.2,
        candidate_top_k: int = 20,
        max_chars_per_evidence: int = 900,
    ):
        self.max_evidence_per_field = top_k_per_query
        self.candidate_top_k = candidate_top_k
        self.max_chars_per_evidence = max_chars_per_evidence

        self.evidence_builder = EvidenceBuilder()

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
            logger.info("Generating v2 research gap profile for: %s", file_name)

            evidence_packs = self.evidence_builder.build_multiple_evidence_packs(
                file_name=file_name,
                field_names=GAP_EVIDENCE_FIELDS,
                candidate_top_k=self.candidate_top_k,
                max_evidence_chunks=self.max_evidence_per_field,
                max_chars_per_chunk=self.max_chars_per_evidence,
            )

            evidence_context, sources, source_counter = self._format_evidence_context(
                evidence_packs=evidence_packs,
                start_index=source_counter,
            )

            all_sources.extend(sources)

            profile, raw_output = self._generate_single_paper_gap_profile(
                file_name=file_name,
                evidence_context=evidence_context,
            )

            paper_profiles.append(profile.to_dict())
            raw_outputs.append(
                f"\n\n--- RAW GAP PROFILE FOR {file_name} ---\n{raw_output}"
            )

        cross_report, cross_raw_output = self._generate_cross_paper_gap_report(
            paper_profiles=paper_profiles,
            file_names=file_names,
        )

        raw_outputs.append(
            "\n\n--- RAW CROSS-PAPER GAP OUTPUT ---\n" + cross_raw_output
        )

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

    def _generate_single_paper_gap_profile(
        self,
        file_name: str,
        evidence_context: str,
    ) -> tuple[PaperGapProfile, str]:
        prompt = self._build_single_paper_gap_prompt(
            file_name=file_name,
            evidence_context=evidence_context,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        parsed_output, parse_log = self._parse_or_repair_json(
            raw_output=raw_output,
            repair_builder=lambda cleaned: self._repair_gap_profile_to_json(
                raw_output=cleaned,
                file_name=file_name,
            ),
        )

        raw_trace = raw_output + parse_log

        profile = PaperGapProfile(
            paper=str(parsed_output.get("paper", file_name)),
            problem_area=str(
                parsed_output.get("problem_area", "Not found in retrieved context.")
            ),
            stated_or_implied_research_gap=str(
                parsed_output.get(
                    "stated_or_implied_research_gap",
                    "Not found in retrieved context.",
                )
            ),
            dataset_gaps=str(
                parsed_output.get("dataset_gaps", "Not found in retrieved context.")
            ),
            evaluation_gaps=str(
                parsed_output.get("evaluation_gaps", "Not found in retrieved context.")
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
            limitations=str(
                parsed_output.get("limitations", "Not found in retrieved context.")
            ),
            future_work=str(
                parsed_output.get("future_work", "Not found in retrieved context.")
            ),
            possible_research_opportunities=self._ensure_string_list(
                parsed_output.get("possible_research_opportunities", [])
            ),
        )

        return profile, raw_trace

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
        parsed_output, parse_log = self._parse_or_repair_json(
            raw_output=raw_output,
            repair_builder=lambda cleaned: self._repair_cross_gap_report_to_json(
                raw_output=cleaned,
                file_names=file_names,
            ),
        )

        return parsed_output, raw_output + parse_log

    def _format_evidence_context(
        self,
        evidence_packs: dict[str, EvidencePack],
        start_index: int,
    ) -> tuple[str, list[dict], int]:
        context_blocks = []
        sources = []

        chunk_id_to_source_id: dict[str, str] = {}
        chunk_id_to_fields: dict[str, set[str]] = {}

        source_counter = start_index

        for field_name, pack in evidence_packs.items():
            context_blocks.append("\n" + "=" * 80)
            context_blocks.append(f"EVIDENCE FIELD: {field_name}")
            context_blocks.append(f"FIELD QUERY: {pack.query}")
            context_blocks.append("=" * 80)

            if not pack.evidence_chunks:
                context_blocks.append(
                    "No strong evidence was selected for this field."
                )
                continue

            for chunk in pack.evidence_chunks:
                if chunk.chunk_id not in chunk_id_to_source_id:
                    source_id = f"Source {source_counter}"
                    chunk_id_to_source_id[chunk.chunk_id] = source_id
                    chunk_id_to_fields[chunk.chunk_id] = set()
                    source_counter += 1

                    sources.append(
                        {
                            "source_id": source_id,
                            "paper_title": chunk.paper_title,
                            "file_name": chunk.file_name,
                            "page_number": chunk.page_number,
                            "section_title": chunk.section_title,
                            "citation": chunk.source,
                            "chunk_id": chunk.chunk_id,
                            "final_score": chunk.final_score,
                            "dense_score": chunk.dense_score,
                            "keyword_score": chunk.keyword_score,
                            "section_score": chunk.section_score,
                            "page_score": chunk.page_score,
                            "noise_penalty": chunk.noise_penalty,
                            "used_for_fields": [],
                        }
                    )

                source_id = chunk_id_to_source_id[chunk.chunk_id]
                chunk_id_to_fields[chunk.chunk_id].add(field_name)

                context_blocks.append(
                    (
                        f"\n[{source_id}]\n"
                        f"File: {chunk.file_name}\n"
                        f"Paper: {chunk.paper_title}\n"
                        f"Page: {chunk.page_number}\n"
                        f"Section: {chunk.section_title}\n"
                        f"Evidence Field: {field_name}\n"
                        f"Evidence Score: {chunk.final_score:.4f}\n"
                        f"Text:\n{chunk.text_preview}\n"
                    )
                )

        for source in sources:
            chunk_id = source["chunk_id"]
            source["used_for_fields"] = sorted(
                list(chunk_id_to_fields.get(chunk_id, []))
            )

        return "\n".join(context_blocks), sources, source_counter

    def _build_single_paper_gap_prompt(
        self,
        file_name: str,
        evidence_context: str,
    ) -> str:
        return f"""
You are PaperMind Research Gap Finder v2.

Your task is to generate a structured research-gap profile for exactly one paper.

Selected file:
{file_name}

Important rules:
1. Use only the provided evidence context.
2. Do not invent unsupported facts.
3. Every important claim should cite source labels such as [Source 1], [Source 2].
4. A research gap does not need an exact heading called "research gap".
5. If a gap is implicit in the abstract, introduction, related work, results, limitations, or future work, infer it carefully and cite the supporting evidence.
6. Separate stated limitations from inferred opportunities.
7. Only write "Not found in retrieved context" if no supporting evidence exists.
8. Return valid JSON only.
9. Do not use markdown.
10. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "paper": "{file_name}",
  "problem_area": "...",
  "stated_or_implied_research_gap": "...",
  "dataset_gaps": "...",
  "evaluation_gaps": "...",
  "scalability_gaps": "...",
  "robustness_gaps": "...",
  "reproducibility_gaps": "...",
  "limitations": "...",
  "future_work": "...",
  "possible_research_opportunities": [
    "..."
  ]
}}

Evidence Context:
{evidence_context}
""".strip()

    def _build_cross_paper_gap_prompt(
        self,
        paper_profiles: list[dict],
        file_names: list[str],
    ) -> str:
        selected_files = ", ".join(file_names)
        profiles_json = json.dumps(paper_profiles, indent=2, ensure_ascii=False)

        return f"""
You are PaperMind, acting as a research supervisor.

Your task is to synthesize research gaps across multiple structured paper profiles.

Selected files:
{selected_files}

Rules:
1. Use only the structured paper profiles below.
2. Do not invent unsupported details.
3. Focus on actionable research opportunities.
4. If only one paper is provided, generate a single-paper synthesis.
5. Return valid JSON only.
6. Do not use markdown.
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
  "stated_or_implied_research_gap": "...",
  "dataset_gaps": "...",
  "evaluation_gaps": "...",
  "scalability_gaps": "...",
  "robustness_gaps": "...",
  "reproducibility_gaps": "...",
  "limitations": "...",
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

    def _parse_or_repair_json(
        self,
        raw_output: str,
        repair_builder,
    ) -> tuple[dict, str]:
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            return self._parse_json_output(cleaned_output), ""

        except ValueError:
            logger.warning("Model output was not valid JSON. Attempting repair.")

            repaired_output = repair_builder(cleaned_output)
            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                return (
                    self._parse_json_output(repaired_cleaned),
                    "\n\n--- JSON REPAIR OUTPUT ---\n\n" + repaired_output,
                )

            except ValueError:
                logger.warning("JSON repair failed. Using fallback JSON.")
                return (
                    self._fallback_json(),
                    "\n\n--- JSON REPAIR FAILED ---\n\n" + repaired_output,
                )

    @staticmethod
    def _fallback_json() -> dict:
        return {
            "paper": "Not reliably extracted.",
            "problem_area": "Not reliably extracted.",
            "stated_or_implied_research_gap": "Not reliably extracted.",
            "dataset_gaps": "Not reliably extracted.",
            "evaluation_gaps": "Not reliably extracted.",
            "scalability_gaps": "Not reliably extracted.",
            "robustness_gaps": "Not reliably extracted.",
            "reproducibility_gaps": "Not reliably extracted.",
            "limitations": "Not reliably extracted.",
            "future_work": "Not reliably extracted.",
            "possible_research_opportunities": [
                "Not reliably extracted."
            ],
            "common_limitations": [
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
            "overall_summary": "Structured gap report could not be reliably generated.",
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
    def _ensure_string_list(value: Any) -> list[str]:
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
        "| Paper | Problem Area | Research Gap | Dataset Gaps | Evaluation Gaps | Scalability Gaps | Robustness Gaps | Reproducibility Gaps | Limitations | Future Work |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for profile in report.paper_gap_profiles:
        lines.append(
            "| "
            + clean_markdown_cell(profile.get("paper", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("problem_area", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("stated_or_implied_research_gap", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("dataset_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("evaluation_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("scalability_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("robustness_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("reproducibility_gaps", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("limitations", "Not found"))
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
        fields = source.get("used_for_fields", [])

        if fields:
            field_text = ", ".join(fields)
        else:
            field_text = "unknown"

        lines.append(
            f"- **{source['source_id']}**: "
            f"{source['citation']} "
            f"(score={source.get('final_score', 0):.4f}, fields={field_text})"
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
        help="Compact evidence chunks per evidence field.",
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=20,
        help="Number of dense retrieval candidates inspected internally.",
    )

    parser.add_argument(
        "--max-chars-per-evidence",
        type=int,
        default=900,
        help="Maximum characters per evidence chunk.",
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
        candidate_top_k=args.candidate_top_k,
        max_chars_per_evidence=args.max_chars_per_evidence,
    )

    report = finder.generate(
        file_names=args.file_names,
    )

    report_dict = report.to_dict()
    markdown = gap_report_to_markdown(report)

    print("\nRESEARCH GAP REPORT V2")
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