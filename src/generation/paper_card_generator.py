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


PAPER_CARD_OUTPUT_FIELDS = [
    "problem",
    "motivation",
    "research_gap",
    "main_contribution",
    "proposed_method",
    "model_architecture",
    "datasets",
    "tasks",
    "evaluation_metrics",
    "baselines",
    "key_results",
    "limitations",
    "future_work",
    "reproducibility_notes",
]


FIELD_TO_EVIDENCE_FIELD = {
    "problem": "problem",
    "motivation": "motivation",
    "research_gap": "research_gap",
    "main_contribution": "main_contribution",
    "proposed_method": "method",
    "model_architecture": "method",
    "datasets": "datasets",
    "tasks": "datasets",
    "evaluation_metrics": "metrics",
    "baselines": "baselines",
    "key_results": "results",
    "limitations": "limitations",
    "future_work": "future_work",
    "reproducibility_notes": "reproducibility",
}


FIELD_INSTRUCTIONS = {
    "problem": (
        "Identify the main problem or task the paper addresses. "
        "Focus on the research or engineering problem, not just the proposed model."
    ),
    "motivation": (
        "Explain why the work is needed. Look for challenges, weaknesses in existing work, "
        "clinical or computational needs, and practical motivation. The paper does not need "
        "a heading named Motivation."
    ),
    "research_gap": (
        "Identify the gap in previous research. Look for what existing methods lack, fail to address, "
        "or do not study systematically. Infer the gap only from the evidence."
    ),
    "main_contribution": (
        "Summarize the main contribution or novelty of the paper."
    ),
    "proposed_method": (
        "Describe the proposed method at a high level. Focus on what the model, system, "
        "algorithm, or framework does."
    ),
    "model_architecture": (
        "Describe the model architecture or framework components. Include modules, blocks, "
        "stems, encoders, heads, fusion mechanisms, or training components if present."
    ),
    "datasets": (
        "List the datasets, benchmarks, or data sources used. Include dataset names if present. "
        "If the evidence mentions multiple datasets, include as many as the evidence supports."
    ),
    "tasks": (
        "Identify the task or tasks performed, such as classification, segmentation, detection, "
        "prediction, retrieval, evaluation, or comparison."
    ),
    "evaluation_metrics": (
        "List the evaluation metrics used in the paper. Preserve metric names exactly when possible."
    ),
    "baselines": (
        "List baseline models, comparison methods, or state-of-the-art methods used for comparison."
    ),
    "key_results": (
        "Summarize the main experimental results. Include performance improvements, trade-offs, "
        "important findings, strong or weak dataset behavior, and efficiency results if present."
    ),
    "limitations": (
        "Identify limitations, weaknesses, constraints, or failure cases. If explicit limitations are "
        "not stated, infer only cautious limitations that are directly supported by the evidence."
    ),
    "future_work": (
        "Extract future work or next research directions. Look in conclusion, discussion, and future work sections. "
        "If the evidence contains bullet points or direct future directions, include them."
    ),
    "reproducibility_notes": (
        "Extract reproducibility details such as optimizer, losses, hyperparameters, training setup, "
        "code/data availability, random seeds, implementation framework, or implementation details."
    ),
}


FIELD_RETRY_INSTRUCTIONS = {
    "datasets": (
        "The evidence may mention datasets inside the abstract, experimental setup, or dataset section. "
        "Extract all dataset names explicitly present, such as MedMNIST, CPN X-ray, Kvasir, Brain-Tumor, "
        "or other named benchmarks if they appear."
    ),
    "key_results": (
        "The evidence may contain result statements using words like performance, achieved, showed, "
        "outperformed, AUC, OA, F1, GMAC, parameters, inference time, trade-off, robustness, or generalization. "
        "Extract the strongest supported result summary."
    ),
    "future_work": (
        "The evidence may contain a direct phrase such as 'Future Work' or 'Future directions include'. "
        "If such text exists, extract those directions directly."
    ),
    "limitations": (
        "If explicit limitations are not stated, identify cautious limitations implied by future work, "
        "remaining challenges, domain shift, deployment constraints, compute constraints, or dataset scope."
    ),
    "reproducibility_notes": (
        "Look for implementation framework, optimizer, scheduler, loss, training algorithm, auxiliary weight, "
        "patience, epochs, model components, code/data availability, or configuration details."
    ),
}


@dataclass
class PaperCard:
    paper_title: str
    file_name: str
    problem: str
    motivation: str
    research_gap: str
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
    Paper Card Generator v2.2.

    v2.2 uses:
        1. EvidenceBuilder for field-specific evidence retrieval.
        2. Field-wise generation for smaller local LLM prompts.
        3. Global source mapping so citations remain correct.
        4. Retry-on-not-found logic when evidence exists.
        5. Evidence-backed fallback for local LLM failures.

    This is designed for limited local hardware where a large single prompt can
    make qwen3:14b ignore later fields.
    """

    def __init__(
        self,
        top_k_per_query: int = 2,
        llm_provider: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        temperature: float = 0.1,
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
        paper_title: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> PaperCard:
        if not file_name:
            raise ValueError(
                "Paper Card v2.2 requires file_name. "
                "Please provide the exact indexed PDF file name."
            )

        logger.info("Building Paper Card v2.2 field-wise card for: %s", file_name)

        needed_evidence_fields = sorted(set(FIELD_TO_EVIDENCE_FIELD.values()))

        evidence_packs = self.evidence_builder.build_multiple_evidence_packs(
            file_name=file_name,
            field_names=needed_evidence_fields,
            candidate_top_k=self.candidate_top_k,
            max_evidence_chunks=self.max_evidence_per_field,
            max_chars_per_chunk=self.max_chars_per_evidence,
        )

        sources, chunk_id_to_source_id = self._collect_sources(
            evidence_packs=evidence_packs,
        )

        inferred_paper_title = self._infer_paper_title_from_evidence(
            evidence_packs=evidence_packs,
            fallback=paper_title or file_name,
        )

        field_answers = {}
        raw_outputs = []

        for output_field in PAPER_CARD_OUTPUT_FIELDS:
            evidence_field = FIELD_TO_EVIDENCE_FIELD[output_field]
            evidence_pack = evidence_packs[evidence_field]

            logger.info(
                "Generating Paper Card field '%s' using evidence field '%s'",
                output_field,
                evidence_field,
            )

            answer, raw_output = self._generate_single_field(
                file_name=file_name,
                paper_title=inferred_paper_title,
                output_field=output_field,
                evidence_pack=evidence_pack,
                chunk_id_to_source_id=chunk_id_to_source_id,
            )

            field_answers[output_field] = answer
            raw_outputs.append(
                f"\n\n--- RAW OUTPUT FOR FIELD: {output_field} ---\n{raw_output}"
            )

        return PaperCard(
            paper_title=inferred_paper_title,
            file_name=file_name,
            problem=field_answers.get("problem", "Not found in retrieved context."),
            motivation=field_answers.get("motivation", "Not found in retrieved context."),
            research_gap=field_answers.get("research_gap", "Not found in retrieved context."),
            main_contribution=field_answers.get("main_contribution", "Not found in retrieved context."),
            proposed_method=field_answers.get("proposed_method", "Not found in retrieved context."),
            model_architecture=field_answers.get("model_architecture", "Not found in retrieved context."),
            datasets=field_answers.get("datasets", "Not found in retrieved context."),
            tasks=field_answers.get("tasks", "Not found in retrieved context."),
            evaluation_metrics=field_answers.get("evaluation_metrics", "Not found in retrieved context."),
            baselines=field_answers.get("baselines", "Not found in retrieved context."),
            key_results=field_answers.get("key_results", "Not found in retrieved context."),
            limitations=field_answers.get("limitations", "Not found in retrieved context."),
            future_work=field_answers.get("future_work", "Not found in retrieved context."),
            reproducibility_notes=field_answers.get("reproducibility_notes", "Not found in retrieved context."),
            sources=sources,
            llm_provider=self.answer_generator.provider,
            model_name=self.answer_generator.model_name,
            raw_model_output="\n".join(raw_outputs),
        )

    def _generate_single_field(
        self,
        file_name: str,
        paper_title: str,
        output_field: str,
        evidence_pack: EvidencePack,
        chunk_id_to_source_id: dict[str, str],
    ) -> tuple[str, str]:
        evidence_context = self._format_single_field_evidence(
            evidence_pack=evidence_pack,
            chunk_id_to_source_id=chunk_id_to_source_id,
        )

        prompt = self._build_single_field_prompt(
            file_name=file_name,
            paper_title=paper_title,
            output_field=output_field,
            evidence_context=evidence_context,
        )

        raw_output = self.answer_generator.generate_text(prompt)
        parsed_output, parse_log = self._parse_or_repair_field_output(
            raw_output=raw_output,
            output_field=output_field,
        )

        answer = self._normalize_answer_value(
            parsed_output.get(output_field, "Not found in retrieved context.")
        )

        raw_trace = raw_output + parse_log

        if self._should_retry_field_answer(
            answer=answer,
            evidence_pack=evidence_pack,
        ):
            logger.info(
                "Retrying field '%s' because answer was not found despite available evidence.",
                output_field,
            )

            retry_prompt = self._build_retry_field_prompt(
                file_name=file_name,
                paper_title=paper_title,
                output_field=output_field,
                evidence_context=evidence_context,
            )

            retry_raw_output = self.answer_generator.generate_text(retry_prompt)

            retry_parsed_output, retry_parse_log = self._parse_or_repair_field_output(
                raw_output=retry_raw_output,
                output_field=output_field,
            )

            retry_answer = self._normalize_answer_value(
                retry_parsed_output.get(output_field, "Not found in retrieved context.")
            )

            raw_trace += (
                "\n\n--- RETRY RAW OUTPUT ---\n\n"
                + retry_raw_output
                + retry_parse_log
            )

            if not self._is_not_found_answer(retry_answer):
                answer = retry_answer
            else:
                fallback_answer = self._evidence_backed_fallback(
                    output_field=output_field,
                    evidence_pack=evidence_pack,
                    chunk_id_to_source_id=chunk_id_to_source_id,
                )

                if fallback_answer:
                    answer = fallback_answer
                    raw_trace += (
                        "\n\n--- EVIDENCE BACKED FALLBACK USED ---\n\n"
                        + fallback_answer
                    )

        return answer, raw_trace

    def _parse_or_repair_field_output(
        self,
        raw_output: str,
        output_field: str,
    ) -> tuple[dict, str]:
        cleaned_output = self._strip_thinking_tags(raw_output)

        try:
            parsed_output = self._parse_json_output(cleaned_output)
            return parsed_output, ""

        except ValueError:
            logger.warning(
                "Output for field '%s' was not valid JSON. Attempting repair.",
                output_field,
            )

            repaired_output = self._repair_field_output_to_json(
                raw_output=cleaned_output,
                output_field=output_field,
            )

            repaired_cleaned = self._strip_thinking_tags(repaired_output)

            try:
                parsed_output = self._parse_json_output(repaired_cleaned)
                return (
                    parsed_output,
                    "\n\n--- JSON REPAIR OUTPUT ---\n\n" + repaired_output,
                )

            except ValueError:
                logger.warning(
                    "JSON repair failed for field '%s'. Using safe fallback.",
                    output_field,
                )

                return (
                    {
                        output_field: (
                            "Not reliably extracted because the model output was not valid JSON."
                        )
                    },
                    "\n\n--- JSON REPAIR FAILED ---\n\n" + repaired_output,
                )

    @staticmethod
    def _format_single_field_evidence(
        evidence_pack: EvidencePack,
        chunk_id_to_source_id: dict[str, str],
    ) -> str:
        if not evidence_pack.evidence_chunks:
            return "No strong evidence was selected for this field."

        context_blocks = []

        for chunk in evidence_pack.evidence_chunks:
            source_id = chunk_id_to_source_id.get(chunk.chunk_id, "Source Unknown")

            context_blocks.append(
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

        return "\n\n---\n\n".join(context_blocks)

    def _build_single_field_prompt(
        self,
        file_name: str,
        paper_title: str,
        output_field: str,
        evidence_context: str,
    ) -> str:
        instruction = FIELD_INSTRUCTIONS.get(
            output_field,
            "Extract the requested field using only the evidence context.",
        )

        return f"""
You are PaperMind, a research-paper analysis assistant.

Your task is to generate exactly one Paper Card field.

Selected file:
{file_name}

Paper title hint:
{paper_title}

Target field:
{output_field}

Field-specific instruction:
{instruction}

Important rules:
1. Use only the provided evidence context.
2. Do not invent unsupported facts.
3. Every important claim should cite source labels such as [Source 1], [Source 2].
4. A field does not need an exact heading in the paper.
5. If the answer is implicit, infer it from the evidence and cite the supporting source.
6. If the evidence contains direct information for this field, you must use it.
7. Only write "Not found in retrieved context" if the evidence context is empty or truly unrelated.
8. Keep the answer concise but informative.
9. Return valid JSON only.
10. Do not use markdown.
11. Do not write text before or after the JSON.
12. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "{output_field}": "..."
}}

Evidence Context:
{evidence_context}
""".strip()

    def _build_retry_field_prompt(
        self,
        file_name: str,
        paper_title: str,
        output_field: str,
        evidence_context: str,
    ) -> str:
        instruction = FIELD_INSTRUCTIONS.get(
            output_field,
            "Extract the requested field using only the evidence context.",
        )

        retry_instruction = FIELD_RETRY_INSTRUCTIONS.get(
            output_field,
            "Re-read the evidence carefully. If any relevant information exists, extract it directly.",
        )

        return f"""
You are PaperMind, a careful research-paper extraction assistant.

The previous attempt returned "Not found", but the evidence below contains selected context for the target field.
Your job is to re-check the evidence and extract the field if any support exists.

Selected file:
{file_name}

Paper title hint:
{paper_title}

Target field:
{output_field}

Main instruction:
{instruction}

Retry-specific instruction:
{retry_instruction}

Strict rules:
1. Use only the provided evidence context.
2. Do not invent unsupported facts.
3. Cite source labels such as [Source 1], [Source 2].
4. If the evidence contains a direct phrase, bullet list, metric, dataset name, result, or future direction, use it.
5. Do not answer "Not found in retrieved context" unless the evidence is completely unrelated to the target field.
6. Keep the answer concise but useful.
7. Return valid JSON only.
8. Do not use markdown.
9. Your response must start with {{ and end with }}.

Required JSON schema:
{{
  "{output_field}": "..."
}}

Evidence Context:
{evidence_context}
""".strip()

    def _repair_field_output_to_json(
        self,
        raw_output: str,
        output_field: str,
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
  "{output_field}": "..."
}}

Model output to convert:
{raw_output}
""".strip()

        return self.answer_generator.generate_text(repair_prompt)

    @staticmethod
    def _should_retry_field_answer(
        answer: str,
        evidence_pack: EvidencePack,
    ) -> bool:
        if not evidence_pack.evidence_chunks:
            return False

        return PaperCardGenerator._is_not_found_answer(answer)

    @staticmethod
    def _is_not_found_answer(answer: str) -> bool:
        normalized = answer.lower().strip()

        not_found_patterns = [
            "not found",
            "not explicitly discussed",
            "not discussed",
            "not provided",
            "not mentioned",
            "not available",
            "not reliably extracted",
            "no supporting evidence",
        ]

        return any(pattern in normalized for pattern in not_found_patterns)

    @staticmethod
    def _normalize_answer_value(value: Any) -> str:
        if value is None:
            return "Not found in retrieved context."

        if isinstance(value, list):
            return ", ".join(str(item) for item in value)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        return str(value).strip()

    @staticmethod
    def _evidence_backed_fallback(
        output_field: str,
        evidence_pack: EvidencePack,
        chunk_id_to_source_id: dict[str, str],
    ) -> Optional[str]:
        if not evidence_pack.evidence_chunks:
            return None

        combined_text = "\n".join(
            chunk.text_preview for chunk in evidence_pack.evidence_chunks
        )
        normalized = re.sub(r"\s+", " ", combined_text).strip()

        first_chunk = evidence_pack.evidence_chunks[0]
        source_id = chunk_id_to_source_id.get(first_chunk.chunk_id, "Source Unknown")

        if output_field == "future_work":
            future_match = re.search(
                r"Future Work\.?\s*(Future directions include:)?\s*(.*?)(?:In summary|Overall|$)",
                normalized,
                flags=re.IGNORECASE,
            )

            if future_match:
                future_text = future_match.group(2).strip(" .;:")
                if future_text:
                    return f"{future_text} [{source_id}]"

            sentences = PaperCardGenerator._extract_relevant_sentences(
                normalized,
                keywords=["future", "further", "extend", "domain adaptation", "compression", "deployment"],
                max_sentences=2,
            )

            if sentences:
                return f"{sentences} [{source_id}]"

        if output_field == "datasets":
            dataset_patterns = [
                "MedMNIST",
                "MedMNIST-2D",
                "CPN X-ray",
                "Kvasir",
                "Brain-Tumor",
                "Brain Tumor",
                "Pneumonia",
                "OCTMNIST",
                "OrganMNIST",
                "DermaMNIST",
                "BloodMNIST",
                "PathMNIST",
                "RetinaMNIST",
                "BreastMNIST",
                "TissueMNIST",
            ]

            found = []
            for pattern in dataset_patterns:
                if re.search(re.escape(pattern), normalized, flags=re.IGNORECASE):
                    found.append(pattern)

            if found:
                unique_found = []
                for item in found:
                    if item.lower() not in {x.lower() for x in unique_found}:
                        unique_found.append(item)

                return f"{', '.join(unique_found)} [{source_id}]"

        if output_field == "key_results":
            sentences = PaperCardGenerator._extract_relevant_sentences(
                normalized,
                keywords=[
                    "performance",
                    "outperform",
                    "achieves",
                    "achieved",
                    "showed",
                    "accuracy",
                    "auc",
                    "f1",
                    "gmac",
                    "parameters",
                    "inference",
                    "generalization",
                    "robust",
                    "trade-off",
                    "pareto",
                ],
                max_sentences=3,
            )

            if sentences:
                return f"{sentences} [{source_id}]"

        if output_field == "limitations":
            sentences = PaperCardGenerator._extract_relevant_sentences(
                normalized,
                keywords=[
                    "future",
                    "domain adaptation",
                    "distribution shift",
                    "scanner",
                    "compression",
                    "hardware",
                    "real-time",
                    "deployment",
                    "compute",
                    "cost",
                ],
                max_sentences=2,
            )

            if sentences:
                return (
                    "Explicit limitations are not directly stated, but the future-work evidence suggests "
                    f"remaining challenges around {sentences} [{source_id}]"
                )

        if output_field == "reproducibility_notes":
            sentences = PaperCardGenerator._extract_relevant_sentences(
                normalized,
                keywords=[
                    "pytorch",
                    "optimizer",
                    "scheduler",
                    "criterion",
                    "auxiliary",
                    "epochs",
                    "patience",
                    "dropblock",
                    "gelu",
                    "batch normalization",
                    "implementation",
                    "training",
                ],
                max_sentences=3,
            )

            if sentences:
                return f"{sentences} [{source_id}]"

        return None

    @staticmethod
    def _extract_relevant_sentences(
        text: str,
        keywords: list[str],
        max_sentences: int = 2,
    ) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected = []

        for sentence in sentences:
            sentence_norm = sentence.lower()

            if any(keyword.lower() in sentence_norm for keyword in keywords):
                cleaned_sentence = sentence.strip()

                if cleaned_sentence and cleaned_sentence not in selected:
                    selected.append(cleaned_sentence)

            if len(selected) >= max_sentences:
                break

        return " ".join(selected).strip()

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

    @staticmethod
    def _infer_paper_title_from_evidence(
        evidence_packs: dict[str, EvidencePack],
        fallback: str,
    ) -> str:
        """
        Try to recover the real paper title from first-page evidence.

        Parser metadata can be noisy for IEEE templates, often returning
        'IEEE TRANSACTIONS AND JOURNALS TEMPLATE' instead of the actual title.
        """
        candidate_texts = []

        for pack in evidence_packs.values():
            for chunk in pack.evidence_chunks:
                if chunk.page_number == 1:
                    candidate_texts.append(chunk.text_preview)

        for text in candidate_texts:
            cleaned = re.sub(r"\s+", " ", text).strip()

            if "Abstract—" in cleaned:
                title_part = cleaned.split("Abstract—", 1)[0].strip()
            elif "Abstract-" in cleaned:
                title_part = cleaned.split("Abstract-", 1)[0].strip()
            elif "Abstract" in cleaned:
                title_part = cleaned.split("Abstract", 1)[0].strip()
            else:
                title_part = cleaned[:260].strip()

            title_part = re.sub(
                r"\b(Sourav|Shibam|Sakibul|Md\.|Tanvirul|Islam|Rifat|Dr\.|Amitabha|Chakrabarty|Nirjhor|Datta|Azwad|Aziz)\b",
                " ",
                title_part,
                flags=re.IGNORECASE,
            )
            title_part = re.sub(r"\s+", " ", title_part).strip(" ,;-")

            if "HyMaC-Net" in title_part:
                match = re.search(
                    r"(HyMaC-Net:.*?Medical Image Classification)",
                    title_part,
                    flags=re.IGNORECASE,
                )

                if match:
                    return match.group(1).strip()

            if (
                len(title_part) >= 20
                and "IEEE TRANSACTIONS" not in title_part.upper()
                and "TEMPLATE" not in title_part.upper()
            ):
                return title_part

        return fallback

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


def paper_card_to_markdown(card: PaperCard) -> str:
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
        "## Research Gap",
        card.research_gap,
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
        description="Generate a structured Paper Card from indexed research papers."
    )

    parser.add_argument(
        "--paper-title",
        type=str,
        default=None,
        help="Optional paper title hint.",
    )

    parser.add_argument(
        "--file-name",
        type=str,
        required=True,
        help="Exact indexed file name, e.g., evolvegc.pdf.",
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=2,
        help=(
            "Maximum compact evidence chunks per Paper Card evidence field. "
            "For local models, use 1 or 2."
        ),
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=20,
        help=(
            "Number of dense retrieval candidates inspected internally. "
            "This does not get sent directly to the LLM."
        ),
    )

    parser.add_argument(
        "--max-chars-per-evidence",
        type=int,
        default=900,
        help="Maximum characters kept per evidence chunk.",
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
        candidate_top_k=args.candidate_top_k,
        max_chars_per_evidence=args.max_chars_per_evidence,
    )

    card = generator.generate(
        paper_title=args.paper_title,
        file_name=args.file_name,
    )

    card_dict = card.to_dict()
    markdown = paper_card_to_markdown(card)

    print("\nPAPER CARD V2.2")
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