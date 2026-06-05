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
from src.utils.logger import get_logger


logger = get_logger(__name__)


COMPARISON_PROFILE_FIELDS = [
    "problem",
    "main_contribution",
    "method",
    "architecture",
    "datasets",
    "tasks",
    "metrics",
    "baselines",
    "key_results",
    "limitations",
    "future_work",
]


FIELD_TO_EVIDENCE_FIELD = {
    "problem": "problem",
    "main_contribution": "main_contribution",
    "method": "method",
    "architecture": "method",
    "datasets": "datasets",
    "tasks": "datasets",
    "metrics": "metrics",
    "baselines": "baselines",
    "key_results": "results",
    "limitations": "limitations",
    "future_work": "future_work",
}


FIELD_INSTRUCTIONS = {
    "problem": (
        "Extract the problem or task the paper addresses. Focus on the research problem, "
        "not only the model name."
    ),
    "main_contribution": (
        "Extract the main contribution, novelty, or central claim of the paper."
    ),
    "method": (
        "Describe the proposed method, model, algorithm, or framework at a high level."
    ),
    "architecture": (
        "Describe important architecture components, modules, or design choices."
    ),
    "datasets": (
        "List datasets, benchmarks, or data sources used by the paper."
    ),
    "tasks": (
        "List the tasks studied, such as node classification, link prediction, image classification, "
        "edge prediction, segmentation, detection, or other tasks."
    ),
    "metrics": (
        "List evaluation metrics used in the paper."
    ),
    "baselines": (
        "List baseline methods, comparison models, or state-of-the-art methods."
    ),
    "key_results": (
        "Summarize key experimental results, improvements, findings, or trade-offs."
    ),
    "limitations": (
        "Extract stated limitations or cautious limitations implied by evidence."
    ),
    "future_work": (
        "Extract future work or next research directions if available."
    ),
}


@dataclass
class PaperComparisonProfile:
    paper: str
    problem: str
    main_contribution: str
    method: str
    architecture: str
    datasets: str
    tasks: str
    metrics: str
    baselines: str
    key_results: str
    limitations: str
    future_work: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiPaperComparison:
    compared_files: list[str]
    comparison_profiles: list[dict]
    cross_paper_summary: str
    key_similarities: list[str]
    key_differences: list[str]
    research_gaps: list[str]
    practical_takeaways: list[str]
    sources: list[dict]
    llm_provider: str
    model_name: str
    raw_model_output: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ComparisonGenerator:
    """
    Multi-Paper Comparison v2.

    v1 compared papers using direct vector retrieval and a large mixed prompt.
    v2 uses EvidenceBuilder and creates one structured profile per paper first.

    This reduces the chance that:
        - one paper dominates the comparison
        - another paper becomes "Not found"
        - unrelated evidence from different papers gets mixed
    """

    def __init__(
        self,
        top_k_per_query: int = 1,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.2,
        candidate_top_k: int = 20,
        max_chars_per_evidence: int = 800,
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
        file_names: list[str],
    ) -> MultiPaperComparison:
        if not file_names or len(file_names) < 2:
            raise ValueError("Please provide at least two file names for comparison.")

        file_names = list(dict.fromkeys(file_names))

        all_profiles = []
        all_sources = []
        raw_outputs = []

        source_counter = 1

        for file_name in file_names:
            logger.info("Generating comparison profile for: %s", file_name)

            profile, sources, source_counter, raw_trace = self._generate_single_paper_profile(
                file_name=file_name,
                source_counter=source_counter,
            )

            all_profiles.append(profile.to_dict())
            all_sources.extend(sources)
            raw_outputs.append(
                f"\n\n--- RAW PROFILE OUTPUT FOR {file_name} ---\n{raw_trace}"
            )

        comparison_summary, raw_summary_output = self._generate_cross_paper_comparison(
            profiles=all_profiles,
            file_names=file_names,
        )

        raw_outputs.append(
            "\n\n--- RAW CROSS-PAPER COMPARISON OUTPUT ---\n"
            + raw_summary_output
        )

        return MultiPaperComparison(
            compared_files=file_names,
            comparison_profiles=all_profiles,
            cross_paper_summary=str(
                comparison_summary.get(
                    "cross_paper_summary",
                    "Not found in retrieved context.",
                )
            ),
            key_similarities=self._ensure_string_list(
                comparison_summary.get("key_similarities", [])
            ),
            key_differences=self._ensure_string_list(
                comparison_summary.get("key_differences", [])
            ),
            research_gaps=self._ensure_string_list(
                comparison_summary.get("research_gaps", [])
            ),
            practical_takeaways=self._ensure_string_list(
                comparison_summary.get("practical_takeaways", [])
            ),
            sources=all_sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output="\n".join(raw_outputs),
        )

    def _generate_single_paper_profile(
        self,
        file_name: str,
        source_counter: int,
    ) -> tuple[PaperComparisonProfile, list[dict], int, str]:
        needed_evidence_fields = sorted(set(FIELD_TO_EVIDENCE_FIELD.values()))

        evidence_packs = self.evidence_builder.build_multiple_evidence_packs(
            file_name=file_name,
            field_names=needed_evidence_fields,
            candidate_top_k=self.candidate_top_k,
            max_evidence_chunks=self.max_evidence_per_field,
            max_chars_per_chunk=self.max_chars_per_evidence,
        )

        sources, chunk_id_to_source_id, next_source_counter = self._collect_sources(
            evidence_packs=evidence_packs,
            start_index=source_counter,
        )

        field_answers = {}
        raw_outputs = []

        for profile_field in COMPARISON_PROFILE_FIELDS:
            evidence_field = FIELD_TO_EVIDENCE_FIELD[profile_field]
            evidence_pack = evidence_packs[evidence_field]

            answer, raw_trace = self._generate_profile_field(
                file_name=file_name,
                profile_field=profile_field,
                evidence_pack=evidence_pack,
                chunk_id_to_source_id=chunk_id_to_source_id,
            )

            field_answers[profile_field] = answer
            raw_outputs.append(
                f"\n\n--- RAW FIELD OUTPUT FOR {file_name} / {profile_field} ---\n{raw_trace}"
            )

        profile = PaperComparisonProfile(
            paper=file_name,
            problem=field_answers.get("problem", "Not found in retrieved context."),
            main_contribution=field_answers.get("main_contribution", "Not found in retrieved context."),
            method=field_answers.get("method", "Not found in retrieved context."),
            architecture=field_answers.get("architecture", "Not found in retrieved context."),
            datasets=field_answers.get("datasets", "Not found in retrieved context."),
            tasks=field_answers.get("tasks", "Not found in retrieved context."),
            metrics=field_answers.get("metrics", "Not found in retrieved context."),
            baselines=field_answers.get("baselines", "Not found in retrieved context."),
            key_results=field_answers.get("key_results", "Not found in retrieved context."),
            limitations=field_answers.get("limitations", "Not found in retrieved context."),
            future_work=field_answers.get("future_work", "Not found in retrieved context."),
        )

        return profile, sources, next_source_counter, "\n".join(raw_outputs)

    def _generate_profile_field(
        self,
        file_name: str,
        profile_field: str,
        evidence_pack: EvidencePack,
        chunk_id_to_source_id: dict[str, str],
    ) -> tuple[str, str]:
        evidence_context = self._format_evidence_context(
            evidence_pack=evidence_pack,
            chunk_id_to_source_id=chunk_id_to_source_id,
        )

        prompt = self._build_profile_field_prompt(
            file_name=file_name,
            profile_field=profile_field,
            evidence_context=evidence_context,
        )

        raw_output = self.answer_generator.generate_text(prompt)

        parsed_output, parse_log = self._parse_or_repair_field_json(
            raw_output=raw_output,
            profile_field=profile_field,
        )

        answer = self._as_string(
            parsed_output.get(
                profile_field,
                "Not found in retrieved context.",
            )
        )

        raw_trace = raw_output + parse_log

        if self._should_retry_answer(answer, evidence_pack):
            logger.info(
                "Retrying comparison field '%s' for paper '%s'.",
                profile_field,
                file_name,
            )

            retry_prompt = self._build_retry_profile_field_prompt(
                file_name=file_name,
                profile_field=profile_field,
                evidence_context=evidence_context,
            )

            retry_raw_output = self.answer_generator.generate_text(retry_prompt)

            retry_parsed, retry_parse_log = self._parse_or_repair_field_json(
                raw_output=retry_raw_output,
                profile_field=profile_field,
            )

            retry_answer = self._as_string(
                retry_parsed.get(
                    profile_field,
                    "Not found in retrieved context.",
                )
            )

            raw_trace += (
                "\n\n--- RETRY RAW OUTPUT ---\n"
                + retry_raw_output
                + retry_parse_log
            )

            if not self._is_not_found_answer(retry_answer):
                answer = retry_answer

        return answer, raw_trace

    def _generate_cross_paper_comparison(
        self,
        profiles: list[dict],
        file_names: list[str],
    ) -> tuple[dict, str]:
        prompt = self._build_cross_paper_prompt(
            profiles=profiles,
            file_names=file_names,
        )

        raw_output = self.answer_generator.generate_text(prompt)

        parsed_output, parse_log = self._parse_or_repair_cross_json(
            raw_output=raw_output,
            file_names=file_names,
        )

        return parsed_output, raw_output + parse_log

    @staticmethod
    def _format_evidence_context(
        evidence_pack: EvidencePack,
        chunk_id_to_source_id: dict[str, str],
    ) -> str:
        if not evidence_pack.evidence_chunks:
            return "No strong evidence was selected for this field."

        blocks = []

        for chunk in evidence_pack.evidence_chunks:
            source_id = chunk_id_to_source_id.get(chunk.chunk_id, "Source Unknown")

            blocks.append(
                (
                    f"[{source_id}]\n"
                    f"File: {chunk.file_name}\n"
                    f"Paper: {chunk.paper_title}\n"
                    f"Page: {chunk.page_number}\n"
                    f"Section: {chunk.section_title}\n"
                    f"Evidence Field: {evidence_pack.field_name}\n"
                    f"Evidence Score: {chunk.final_score:.4f}\n"
                    f"Text:\n{chunk.text_preview}\n"
                )
            )

        return "\n\n---\n\n".join(blocks)

    def _build_profile_field_prompt(
        self,
        file_name: str,
        profile_field: str,
        evidence_context: str,
    ) -> str:
        instruction = FIELD_INSTRUCTIONS.get(
            profile_field,
            "Extract the requested comparison field from the evidence.",
        )

        return f"""
You are PaperMind Multi-Paper Comparison v2.

Your task is to generate exactly one structured comparison-profile field for one paper.

Selected file:
{file_name}

Target field:
{profile_field}

Field instruction:
{instruction}

Rules:
1. Use only the provided evidence context.
2. Do not invent unsupported facts.
3. Every important claim should cite source labels such as [Source 1], [Source 2].
4. A field does not need an exact heading in the paper.
5. If the answer is implicit, infer cautiously from the evidence and cite it.
6. Only write "Not found in retrieved context" if the evidence is empty or truly unrelated.
7. Keep the answer concise but useful.
8. Return valid JSON only.
9. Do not use markdown.
10. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "{profile_field}": "..."
}}

Evidence Context:
{evidence_context}
""".strip()

    def _build_retry_profile_field_prompt(
        self,
        file_name: str,
        profile_field: str,
        evidence_context: str,
    ) -> str:
        instruction = FIELD_INSTRUCTIONS.get(
            profile_field,
            "Extract the requested comparison field from the evidence.",
        )

        return f"""
You are PaperMind Multi-Paper Comparison v2.

The previous answer was missing or weak.
Re-read the evidence carefully and extract the requested field if any support exists.

Selected file:
{file_name}

Target field:
{profile_field}

Field instruction:
{instruction}

Strict rules:
1. Use only the evidence context.
2. Cite source labels such as [Source 1], [Source 2].
3. Do not invent unsupported facts.
4. If the evidence includes datasets, metrics, baselines, results, or future work, extract them directly.
5. Do not write "Not found in retrieved context" unless the evidence is completely unrelated.
6. Return valid JSON only.
7. Do not use markdown.
8. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "{profile_field}": "..."
}}

Evidence Context:
{evidence_context}
""".strip()

    def _build_cross_paper_prompt(
        self,
        profiles: list[dict],
        file_names: list[str],
    ) -> str:
        selected_files = ", ".join(file_names)
        profiles_json = json.dumps(profiles, indent=2, ensure_ascii=False)

        return f"""
You are PaperMind Multi-Paper Comparison v2.

Your task is to compare multiple papers using their structured profiles.

Selected files:
{selected_files}

Rules:
1. Use only the structured paper profiles below.
2. Do not invent details not present in the profiles.
3. Compare papers across problem, contribution, method, datasets, metrics, results, limitations, and future work.
4. If a profile field is missing, acknowledge the limitation instead of hallucinating.
5. Return valid JSON only.
6. Do not use markdown.
7. Your response must start with {{ and end with }}.

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
  ],
  "practical_takeaways": [
    "..."
  ]
}}

Structured paper profiles:
{profiles_json}
""".strip()

    def _parse_or_repair_field_json(
        self,
        raw_output: str,
        profile_field: str,
    ) -> tuple[dict, str]:
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            return self._parse_json_output(cleaned_output), ""

        except ValueError:
            logger.warning(
                "Comparison profile field '%s' output was not valid JSON. Attempting repair.",
                profile_field,
            )

            repaired_output = self._repair_field_to_json(
                raw_output=cleaned_output,
                profile_field=profile_field,
            )

            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                return (
                    self._parse_json_output(repaired_cleaned),
                    "\n\n--- JSON REPAIR OUTPUT ---\n" + repaired_output,
                )

            except ValueError:
                return (
                    {profile_field: "Not reliably extracted."},
                    "\n\n--- JSON REPAIR FAILED ---\n" + repaired_output,
                )

    def _parse_or_repair_cross_json(
        self,
        raw_output: str,
        file_names: list[str],
    ) -> tuple[dict, str]:
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            return self._parse_json_output(cleaned_output), ""

        except ValueError:
            logger.warning("Cross-paper comparison output was not valid JSON. Attempting repair.")

            repaired_output = self._repair_cross_to_json(
                raw_output=cleaned_output,
                file_names=file_names,
            )

            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                return (
                    self._parse_json_output(repaired_cleaned),
                    "\n\n--- JSON REPAIR OUTPUT ---\n" + repaired_output,
                )

            except ValueError:
                return (
                    {
                        "cross_paper_summary": "Not reliably extracted.",
                        "key_similarities": ["Not reliably extracted."],
                        "key_differences": ["Not reliably extracted."],
                        "research_gaps": ["Not reliably extracted."],
                        "practical_takeaways": ["Not reliably extracted."],
                    },
                    "\n\n--- JSON REPAIR FAILED ---\n" + repaired_output,
                )

    def _repair_field_to_json(
        self,
        raw_output: str,
        profile_field: str,
    ) -> str:
        repair_prompt = f"""
Convert the following model output into valid JSON only.

Rules:
1. Your response must start with {{ and end with }}.
2. Do not use markdown.
3. Do not explain anything.
4. If the answer is missing, use "Not found in retrieved context."

Required JSON schema:
{{
  "{profile_field}": "..."
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    def _repair_cross_to_json(
        self,
        raw_output: str,
        file_names: list[str],
    ) -> str:
        selected_files = ", ".join(file_names)

        repair_prompt = f"""
Convert the following model output into valid JSON only.

Selected files:
{selected_files}

Rules:
1. Your response must start with {{ and end with }}.
2. Do not use markdown.
3. Do not explain anything.
4. If a field is missing, use "Not reliably extracted."

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
  ],
  "practical_takeaways": [
    "..."
  ]
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    @staticmethod
    def _should_retry_answer(
        answer: str,
        evidence_pack: EvidencePack,
    ) -> bool:
        if not evidence_pack.evidence_chunks:
            return False

        return ComparisonGenerator._is_not_found_answer(answer)

    @staticmethod
    def _is_not_found_answer(answer: str) -> bool:
        normalized = answer.lower().strip()

        patterns = [
            "not found",
            "not discussed",
            "not provided",
            "not mentioned",
            "not available",
            "not reliably extracted",
            "no supporting evidence",
        ]

        return any(pattern in normalized for pattern in patterns)

    @staticmethod
    def _collect_sources(
        evidence_packs: dict[str, EvidencePack],
        start_index: int = 1,
    ) -> tuple[list[dict], dict[str, str], int]:
        chunk_id_to_source: dict[str, dict] = {}
        chunk_id_to_fields: dict[str, set[str]] = {}
        chunk_id_to_source_id: dict[str, str] = {}

        source_counter = start_index

        for field_name, pack in evidence_packs.items():
            for chunk in pack.evidence_chunks:
                if chunk.chunk_id not in chunk_id_to_source:
                    source_id = f"Source {source_counter}"
                    source_counter += 1

                    chunk_id_to_source_id[chunk.chunk_id] = source_id

                    chunk_id_to_source[chunk.chunk_id] = {
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

                    chunk_id_to_fields[chunk.chunk_id] = set()

                chunk_id_to_fields[chunk.chunk_id].add(field_name)

        sources = list(chunk_id_to_source.values())

        for source in sources:
            chunk_id = source["chunk_id"]
            source["used_for_fields"] = sorted(
                list(chunk_id_to_fields.get(chunk_id, []))
            )

        return sources, chunk_id_to_source_id, source_counter

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

    @staticmethod
    def _as_string(value: Any) -> str:
        if value is None:
            return "Not found in retrieved context."

        if isinstance(value, list):
            return ", ".join(str(item) for item in value)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        return str(value).strip()


def clean_markdown_cell(value: Any) -> str:
    text = str(value)
    text = text.replace("\n", " ")
    text = text.replace("|", "/")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def comparison_to_markdown(comparison: MultiPaperComparison) -> str:
    lines = [
        "# Multi-Paper Comparison Matrix",
        "",
        f"**Compared files:** {', '.join(comparison.compared_files)}",
        f"**LLM Provider:** {comparison.llm_provider}",
        f"**Model:** {comparison.model_name}",
        "",
        "## Comparison Table",
        "",
        "| Paper | Problem | Main Contribution | Method | Architecture | Datasets | Tasks | Metrics | Baselines | Key Results | Limitations | Future Work |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for profile in comparison.comparison_profiles:
        lines.append(
            "| "
            + clean_markdown_cell(profile.get("paper", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("problem", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("main_contribution", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("method", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("architecture", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("datasets", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("tasks", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("metrics", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("baselines", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("key_results", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("limitations", "Not found"))
            + " | "
            + clean_markdown_cell(profile.get("future_work", "Not found"))
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

    for item in comparison.key_similarities:
        lines.append(f"- {item}")

    lines.extend(["", "## Key Differences"])

    for item in comparison.key_differences:
        lines.append(f"- {item}")

    lines.extend(["", "## Research Gaps"])

    for item in comparison.research_gaps:
        lines.append(f"- {item}")

    lines.extend(["", "## Practical Takeaways"])

    for item in comparison.practical_takeaways:
        lines.append(f"- {item}")

    lines.extend(["", "## Sources"])

    for source in comparison.sources:
        fields = source.get("used_for_fields", [])
        field_text = ", ".join(fields) if fields else "unknown"

        lines.append(
            f"- **{source['source_id']}**: "
            f"{source['citation']} "
            f"(score={source.get('final_score', 0):.4f}, fields={field_text})"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-paper comparison matrix from indexed research papers."
    )

    parser.add_argument(
        "--file-names",
        nargs="+",
        required=True,
        help="Exact indexed file names to compare.",
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=1,
        help="Compact evidence chunks per comparison evidence field.",
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
        default=800,
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
        default="reports/comparison.json",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/comparison.md",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generator = ComparisonGenerator(
        top_k_per_query=args.top_k_per_query,
        llm_provider=args.llm_provider,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
        candidate_top_k=args.candidate_top_k,
        max_chars_per_evidence=args.max_chars_per_evidence,
    )

    comparison = generator.generate(
        file_names=args.file_names,
    )

    comparison_dict = comparison.to_dict()
    markdown = comparison_to_markdown(comparison)

    print("\nMULTI-PAPER COMPARISON V2")
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