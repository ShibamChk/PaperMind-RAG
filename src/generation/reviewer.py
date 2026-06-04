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


REVIEW_EVIDENCE_FIELDS = [
    "problem",
    "motivation",
    "research_gap",
    "main_contribution",
    "method",
    "datasets",
    "metrics",
    "baselines",
    "results",
    "limitations",
    "future_work",
    "reproducibility",
]


REVIEW_FIELD_CONFIGS = {
    "review_summary": {
        "output_keys": ["review_summary"],
        "evidence_fields": [
            "problem",
            "motivation",
            "research_gap",
            "main_contribution",
            "method",
            "results",
        ],
        "instruction": (
            "Write a concise reviewer-style summary of the paper. Mention the problem, "
            "main idea, method, and evaluation scope if supported by the evidence."
        ),
        "type": "string",
    },
    "main_contributions": {
        "output_keys": ["main_contributions"],
        "evidence_fields": [
            "main_contribution",
            "method",
            "results",
        ],
        "instruction": (
            "List the main contributions of the paper as reviewer bullet points. "
            "Focus on novelty, method, empirical study, and practical contribution."
        ),
        "type": "list",
    },
    "strengths": {
        "output_keys": ["strengths"],
        "evidence_fields": [
            "main_contribution",
            "method",
            "datasets",
            "metrics",
            "results",
            "baselines",
        ],
        "instruction": (
            "List the strongest aspects of the paper. Consider method design, "
            "multi-dataset evaluation, baselines, metrics, efficiency, and empirical findings."
        ),
        "type": "list",
    },
    "weaknesses": {
        "output_keys": ["weaknesses"],
        "evidence_fields": [
            "limitations",
            "future_work",
            "datasets",
            "results",
            "reproducibility",
            "baselines",
        ],
        "instruction": (
            "List weaknesses or concerns. If explicit weaknesses are not stated, infer cautious "
            "reviewer concerns from future work, evaluation scope, reproducibility gaps, dataset scope, "
            "or missing experiments."
        ),
        "type": "list",
    },
    "novelty_assessment": {
        "output_keys": ["novelty_assessment"],
        "evidence_fields": [
            "research_gap",
            "main_contribution",
            "method",
            "baselines",
        ],
        "instruction": (
            "Assess novelty based only on evidence. Explain whether the contribution appears incremental, "
            "moderately novel, or strong, and cite the source evidence."
        ),
        "type": "string",
    },
    "technical_soundness": {
        "output_keys": ["technical_soundness"],
        "evidence_fields": [
            "method",
            "metrics",
            "results",
            "reproducibility",
        ],
        "instruction": (
            "Assess technical soundness. Consider whether the method description, implementation details, "
            "training design, and evaluation support the claims."
        ),
        "type": "string",
    },
    "experimental_quality": {
        "output_keys": ["experimental_quality"],
        "evidence_fields": [
            "datasets",
            "metrics",
            "baselines",
            "results",
        ],
        "instruction": (
            "Assess the experimental quality. Consider datasets, metrics, baselines, result analysis, "
            "efficiency evaluation, and whether the evaluation is broad enough."
        ),
        "type": "string",
    },
    "missing_experiments": {
        "output_keys": ["missing_experiments"],
        "evidence_fields": [
            "limitations",
            "future_work",
            "datasets",
            "results",
            "baselines",
        ],
        "instruction": (
            "Suggest missing experiments that would strengthen the paper. These can be inferred from "
            "future work, limitations, dataset scope, baseline coverage, domain shift, deployment, "
            "or ablation needs."
        ),
        "type": "list",
    },
    "reproducibility_concerns": {
        "output_keys": ["reproducibility_concerns"],
        "evidence_fields": [
            "reproducibility",
            "method",
            "metrics",
        ],
        "instruction": (
            "List reproducibility concerns or available reproducibility details. Consider optimizer, "
            "hyperparameters, code availability, data availability, training protocol, seeds, and implementation details."
        ),
        "type": "list",
    },
    "questions_for_authors": {
        "output_keys": ["questions_for_authors"],
        "evidence_fields": [
            "limitations",
            "future_work",
            "reproducibility",
            "datasets",
            "results",
            "baselines",
        ],
        "instruction": (
            "Generate constructive questions a reviewer would ask the authors. Questions should be grounded "
            "in missing evidence, limitations, future work, evaluation choices, or reproducibility concerns."
        ),
        "type": "list",
    },
}


FINAL_RECOMMENDATION_OPTIONS = [
    "Strong Accept",
    "Accept",
    "Weak Accept",
    "Borderline",
    "Weak Reject",
    "Reject",
    "Not enough evidence in retrieved context",
]


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
    Reviewer Mode v2.1.

    v2.1 uses:
        1. EvidenceBuilder for field-specific reviewer evidence.
        2. Field-wise review generation.
        3. Global source mapping.
        4. Retry-on-not-found logic.
        5. A separate final recommendation step.

    This is more reliable for local LLMs than asking the model to produce
    the full reviewer report in one large JSON response.
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

        logger.info("Generating Reviewer Mode v2.1 report for: %s", file_name)

        evidence_packs = self.evidence_builder.build_multiple_evidence_packs(
            file_name=file_name,
            field_names=REVIEW_EVIDENCE_FIELDS,
            candidate_top_k=self.candidate_top_k,
            max_evidence_chunks=self.max_evidence_per_field,
            max_chars_per_chunk=self.max_chars_per_evidence,
        )

        sources, chunk_id_to_source_id = self._collect_sources(
            evidence_packs=evidence_packs,
        )

        field_outputs: dict[str, Any] = {}
        raw_outputs = []

        for review_field, config in REVIEW_FIELD_CONFIGS.items():
            logger.info("Generating reviewer field: %s", review_field)

            output, raw_trace = self._generate_review_field(
                file_name=file_name,
                review_field=review_field,
                config=config,
                evidence_packs=evidence_packs,
                chunk_id_to_source_id=chunk_id_to_source_id,
            )

            field_outputs[review_field] = output
            raw_outputs.append(
                f"\n\n--- RAW OUTPUT FOR REVIEW FIELD: {review_field} ---\n{raw_trace}"
            )

        final_decision, final_raw_trace = self._generate_final_recommendation(
            file_name=file_name,
            field_outputs=field_outputs,
        )

        raw_outputs.append(
            "\n\n--- RAW OUTPUT FOR FINAL RECOMMENDATION ---\n"
            + final_raw_trace
        )

        return PaperReview(
            paper=file_name,
            review_summary=self._as_string(
                field_outputs.get("review_summary", "Not found in retrieved context.")
            ),
            main_contributions=self._ensure_string_list(
                field_outputs.get("main_contributions", [])
            ),
            strengths=self._ensure_string_list(
                field_outputs.get("strengths", [])
            ),
            weaknesses=self._ensure_string_list(
                field_outputs.get("weaknesses", [])
            ),
            novelty_assessment=self._as_string(
                field_outputs.get("novelty_assessment", "Not found in retrieved context.")
            ),
            technical_soundness=self._as_string(
                field_outputs.get("technical_soundness", "Not found in retrieved context.")
            ),
            experimental_quality=self._as_string(
                field_outputs.get("experimental_quality", "Not found in retrieved context.")
            ),
            missing_experiments=self._ensure_string_list(
                field_outputs.get("missing_experiments", [])
            ),
            reproducibility_concerns=self._ensure_string_list(
                field_outputs.get("reproducibility_concerns", [])
            ),
            questions_for_authors=self._ensure_string_list(
                field_outputs.get("questions_for_authors", [])
            ),
            final_recommendation=self._as_string(
                final_decision.get(
                    "final_recommendation",
                    "Not enough evidence in retrieved context",
                )
            ),
            recommendation_reason=self._as_string(
                final_decision.get(
                    "recommendation_reason",
                    "Not found in retrieved context.",
                )
            ),
            reviewer_confidence=self._as_string(
                final_decision.get(
                    "reviewer_confidence",
                    "Low",
                )
            ),
            sources=sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output="\n".join(raw_outputs),
        )

    def _generate_review_field(
        self,
        file_name: str,
        review_field: str,
        config: dict,
        evidence_packs: dict[str, EvidencePack],
        chunk_id_to_source_id: dict[str, str],
    ) -> tuple[Any, str]:
        evidence_context = self._format_field_evidence_context(
            evidence_fields=config["evidence_fields"],
            evidence_packs=evidence_packs,
            chunk_id_to_source_id=chunk_id_to_source_id,
        )

        prompt = self._build_review_field_prompt(
            file_name=file_name,
            review_field=review_field,
            config=config,
            evidence_context=evidence_context,
        )

        raw_output = self.answer_generator.generate_text(prompt)

        parsed_output, parse_log = self._parse_or_repair_review_field(
            raw_output=raw_output,
            review_field=review_field,
            config=config,
        )

        output_value = parsed_output.get(
            review_field,
            [] if config["type"] == "list" else "Not found in retrieved context.",
        )

        raw_trace = raw_output + parse_log

        if self._should_retry_output(output_value=output_value, evidence_context=evidence_context):
            logger.info(
                "Retrying reviewer field '%s' because output was weak despite evidence.",
                review_field,
            )

            retry_prompt = self._build_retry_review_field_prompt(
                file_name=file_name,
                review_field=review_field,
                config=config,
                evidence_context=evidence_context,
            )

            retry_raw_output = self.answer_generator.generate_text(retry_prompt)

            retry_parsed, retry_parse_log = self._parse_or_repair_review_field(
                raw_output=retry_raw_output,
                review_field=review_field,
                config=config,
            )

            retry_value = retry_parsed.get(
                review_field,
                [] if config["type"] == "list" else "Not found in retrieved context.",
            )

            raw_trace += (
                "\n\n--- RETRY RAW OUTPUT ---\n"
                + retry_raw_output
                + retry_parse_log
            )

            if not self._is_weak_output(retry_value):
                output_value = retry_value

        return output_value, raw_trace

    def _generate_final_recommendation(
        self,
        file_name: str,
        field_outputs: dict[str, Any],
    ) -> tuple[dict, str]:
        prompt = self._build_final_recommendation_prompt(
            file_name=file_name,
            field_outputs=field_outputs,
        )

        raw_output = self.answer_generator.generate_text(prompt)

        parsed_output, parse_log = self._parse_or_repair_final_recommendation(
            raw_output=raw_output,
            file_name=file_name,
        )

        recommendation = self._as_string(
            parsed_output.get(
                "final_recommendation",
                "Not enough evidence in retrieved context",
            )
        )

        if recommendation not in FINAL_RECOMMENDATION_OPTIONS:
            parsed_output["final_recommendation"] = "Borderline"

        return parsed_output, raw_output + parse_log

    def _format_field_evidence_context(
        self,
        evidence_fields: list[str],
        evidence_packs: dict[str, EvidencePack],
        chunk_id_to_source_id: dict[str, str],
    ) -> str:
        context_blocks = []

        for evidence_field in evidence_fields:
            pack = evidence_packs.get(evidence_field)

            context_blocks.append("\n" + "=" * 80)
            context_blocks.append(f"EVIDENCE FIELD: {evidence_field}")

            if not pack or not pack.evidence_chunks:
                context_blocks.append("No strong evidence was selected for this field.")
                continue

            for chunk in pack.evidence_chunks:
                source_id = chunk_id_to_source_id.get(chunk.chunk_id, "Source Unknown")

                context_blocks.append(
                    (
                        f"\n[{source_id}]\n"
                        f"File: {chunk.file_name}\n"
                        f"Paper: {chunk.paper_title}\n"
                        f"Page: {chunk.page_number}\n"
                        f"Section: {chunk.section_title}\n"
                        f"Evidence Field: {evidence_field}\n"
                        f"Evidence Score: {chunk.final_score:.4f}\n"
                        f"Text:\n{chunk.text_preview}\n"
                    )
                )

        return "\n".join(context_blocks)

    def _build_review_field_prompt(
        self,
        file_name: str,
        review_field: str,
        config: dict,
        evidence_context: str,
    ) -> str:
        schema_value = '["..."]' if config["type"] == "list" else '"..."'

        return f"""
You are PaperMind Reviewer Mode v2.1, acting like a careful AI/ML conference reviewer.

Your task is to generate exactly one reviewer-report field.

Selected file:
{file_name}

Target review field:
{review_field}

Field instruction:
{config["instruction"]}

Important rules:
1. Use only the provided evidence context.
2. Do not invent unsupported facts.
3. Every important claim should cite source labels such as [Source 1], [Source 2].
4. Be critical but fair.
5. If the evidence supports a judgment, make the judgment and cite it.
6. If the field is implicit, infer cautiously from evidence about method, experiments, limitations, future work, or reproducibility.
7. Only write "Not found in retrieved context" if the evidence context is empty or truly unrelated.
8. Return valid JSON only.
9. Do not use markdown.
10. Do not write text before or after the JSON.
11. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "{review_field}": {schema_value}
}}

Evidence Context:
{evidence_context}
""".strip()

    def _build_retry_review_field_prompt(
        self,
        file_name: str,
        review_field: str,
        config: dict,
        evidence_context: str,
    ) -> str:
        schema_value = '["..."]' if config["type"] == "list" else '"..."'

        return f"""
You are PaperMind Reviewer Mode v2.1.

The previous attempt produced an empty or weak answer.
Re-read the evidence carefully and generate the requested reviewer field if any support exists.

Selected file:
{file_name}

Target review field:
{review_field}

Field instruction:
{config["instruction"]}

Strict retry rules:
1. Use only the evidence context.
2. Cite source labels such as [Source 1], [Source 2].
3. Do not invent facts, but you may make reviewer judgments that are directly grounded in evidence.
4. If future work mentions a direction, it can support missing experiments or weaknesses.
5. If implementation details are incomplete, it can support reproducibility concerns.
6. Do not answer "Not found in retrieved context" unless the evidence is completely unrelated.
7. Return valid JSON only.
8. Do not use markdown.
9. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "{review_field}": {schema_value}
}}

Evidence Context:
{evidence_context}
""".strip()

    def _build_final_recommendation_prompt(
        self,
        file_name: str,
        field_outputs: dict[str, Any],
    ) -> str:
        review_json = json.dumps(field_outputs, indent=2, ensure_ascii=False)

        options_text = "\n".join(f"- {option}" for option in FINAL_RECOMMENDATION_OPTIONS)

        return f"""
You are PaperMind Reviewer Mode v2.1.

Your task is to assign a final reviewer recommendation based only on the structured review fields.

Selected file:
{file_name}

Allowed final recommendations:
{options_text}

Guidance:
- Use Strong Accept only for very strong novelty, strong experiments, and few weaknesses.
- Use Accept or Weak Accept for solid work with some limitations.
- Use Borderline when strengths and weaknesses are balanced.
- Use Weak Reject or Reject when evidence suggests major experimental, novelty, or reproducibility problems.
- Use Not enough evidence in retrieved context if the structured review is mostly missing.

Return valid JSON only.
Do not use markdown.
Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "final_recommendation": "...",
  "recommendation_reason": "...",
  "reviewer_confidence": "High/Medium/Low"
}}

Structured review fields:
{review_json}
""".strip()

    def _parse_or_repair_review_field(
        self,
        raw_output: str,
        review_field: str,
        config: dict,
    ) -> tuple[dict, str]:
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            return self._parse_json_output(cleaned_output), ""

        except ValueError:
            logger.warning(
                "Reviewer field '%s' output was not valid JSON. Attempting repair.",
                review_field,
            )

            repaired_output = self._repair_review_field_to_json(
                raw_output=cleaned_output,
                review_field=review_field,
                output_type=config["type"],
            )

            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                return (
                    self._parse_json_output(repaired_cleaned),
                    "\n\n--- JSON REPAIR OUTPUT ---\n" + repaired_output,
                )

            except ValueError:
                logger.warning(
                    "JSON repair failed for reviewer field '%s'.",
                    review_field,
                )

                fallback_value = [] if config["type"] == "list" else "Not reliably extracted."

                return (
                    {review_field: fallback_value},
                    "\n\n--- JSON REPAIR FAILED ---\n" + repaired_output,
                )

    def _parse_or_repair_final_recommendation(
        self,
        raw_output: str,
        file_name: str,
    ) -> tuple[dict, str]:
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            return self._parse_json_output(cleaned_output), ""

        except ValueError:
            logger.warning("Final recommendation output was not valid JSON. Attempting repair.")

            repaired_output = self._repair_final_recommendation_to_json(
                raw_output=cleaned_output,
                file_name=file_name,
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
                        "final_recommendation": "Not enough evidence in retrieved context",
                        "recommendation_reason": "The final recommendation could not be reliably parsed.",
                        "reviewer_confidence": "Low",
                    },
                    "\n\n--- JSON REPAIR FAILED ---\n" + repaired_output,
                )

    def _repair_review_field_to_json(
        self,
        raw_output: str,
        review_field: str,
        output_type: str,
    ) -> str:
        schema_value = '["..."]' if output_type == "list" else '"..."'

        repair_prompt = f"""
Convert the following model output into valid JSON only.

Rules:
1. Your response must start with {{ and end with }}.
2. Do not use markdown.
3. Do not explain anything.
4. If the answer is missing, use {"[]" if output_type == "list" else '"Not found in retrieved context."'}.

Required JSON schema:
{{
  "{review_field}": {schema_value}
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    def _repair_final_recommendation_to_json(
        self,
        raw_output: str,
        file_name: str,
    ) -> str:
        repair_prompt = f"""
Convert the following model output into valid JSON only.

Selected file:
{file_name}

Rules:
1. Your response must start with {{ and end with }}.
2. Do not use markdown.
3. Do not explain anything.

Required JSON schema:
{{
  "final_recommendation": "...",
  "recommendation_reason": "...",
  "reviewer_confidence": "High/Medium/Low"
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    @staticmethod
    def _should_retry_output(
        output_value: Any,
        evidence_context: str,
    ) -> bool:
        if not evidence_context or "No strong evidence" in evidence_context and len(evidence_context) < 500:
            return False

        return ReviewerMode._is_weak_output(output_value)

    @staticmethod
    def _is_weak_output(output_value: Any) -> bool:
        if output_value is None:
            return True

        if isinstance(output_value, list):
            if len(output_value) == 0:
                return True

            joined = " ".join(str(item) for item in output_value).lower()
            return ReviewerMode._is_not_found_text(joined)

        text = str(output_value).strip()

        if not text:
            return True

        return ReviewerMode._is_not_found_text(text)

    @staticmethod
    def _is_not_found_text(text: str) -> bool:
        normalized = text.lower()

        weak_patterns = [
            "not found",
            "not explicitly discussed",
            "not discussed",
            "not provided",
            "not mentioned",
            "not available",
            "not reliably extracted",
            "no supporting evidence",
        ]

        return any(pattern in normalized for pattern in weak_patterns)

    @staticmethod
    def _collect_sources(
        evidence_packs: dict[str, EvidencePack],
    ) -> tuple[list[dict], dict[str, str]]:
        chunk_id_to_source: dict[str, dict] = {}
        chunk_id_to_fields: dict[str, set[str]] = {}
        chunk_id_to_source_id: dict[str, str] = {}

        source_counter = 1

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

        return sources, chunk_id_to_source_id

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

    @staticmethod
    def _as_string(value: Any) -> str:
        if value is None:
            return "Not found in retrieved context."

        if isinstance(value, list):
            return "; ".join(str(item) for item in value)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        return str(value)


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
        default=1,
        help="Compact evidence chunks per review evidence field.",
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=20,
        help="Dense retrieval candidates inspected internally.",
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
        candidate_top_k=args.candidate_top_k,
        max_chars_per_evidence=args.max_chars_per_evidence,
    )

    review = reviewer.generate(
        file_name=args.file_name,
    )

    review_dict = review.to_dict()
    markdown = review_to_markdown(review)

    print("\nREVIEWER MODE REPORT V2.1")
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