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

from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import ChromaVectorStore
from src.config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class FieldRetrievalConfig:
    field_name: str
    default_query: str
    preferred_section_keywords: list[str]
    positive_keywords: list[str]
    negative_section_keywords: list[str]
    preferred_page_numbers: list[int]


@dataclass
class EvidenceChunk:
    rank: int
    chunk_id: str
    file_name: str
    paper_title: str
    page_number: int
    section_title: str
    source: str
    text: str
    text_preview: str
    dense_score: float
    keyword_score: float
    section_score: float
    page_score: float
    noise_penalty: float
    final_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidencePack:
    file_name: str
    field_name: str
    query: str
    evidence_chunks: list[EvidenceChunk]

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "field_name": self.field_name,
            "query": self.query,
            "evidence_chunks": [
                chunk.to_dict() for chunk in self.evidence_chunks
            ],
        }


FIELD_CONFIGS: dict[str, FieldRetrievalConfig] = {
    "problem": FieldRetrievalConfig(
        field_name="problem",
        default_query=(
            "What problem does this paper solve? What challenge or task is the paper addressing?"
        ),
        preferred_section_keywords=[
            "abstract",
            "introduction",
            "problem",
            "background",
        ],
        positive_keywords=[
            "problem",
            "challenge",
            "task",
            "address",
            "solve",
            "objective",
            "aim",
            "goal",
            "medical image",
            "classification",
            "generalize",
            "generalization",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
            "auc",
            "table",
            "comparison",
        ],
        preferred_page_numbers=[1, 2],
    ),
    "motivation": FieldRetrievalConfig(
        field_name="motivation",
        default_query=(
            "What motivates this work? What need, challenge, limitation, or weakness "
            "in existing work does the paper describe?"
        ),
        preferred_section_keywords=[
            "abstract",
            "introduction",
            "background",
            "literature review",
            "related work",
        ],
        positive_keywords=[
            "motivation",
            "motivated",
            "challenge",
            "need",
            "requires",
            "existing",
            "previous",
            "limited",
            "limitation",
            "however",
            "although",
            "despite",
            "gap",
            "generalize",
            "generalization",
            "computationally",
            "efficient",
            "lightweight",
            "heterogeneous",
            "small datasets",
            "overfitting",
            "clinical",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
            "auc",
            "table",
            "statistical",
            "ablation",
            "comparison",
        ],
        preferred_page_numbers=[1, 2],
    ),
    "research_gap": FieldRetrievalConfig(
        field_name="research_gap",
        default_query=(
            "What research gap does this paper address? What is missing, limited, "
            "weak, or unexplored in previous methods?"
        ),
        preferred_section_keywords=[
            "abstract",
            "introduction",
            "literature review",
            "related work",
            "background",
        ],
        positive_keywords=[
            "gap",
            "limited",
            "limitation",
            "weakness",
            "missing",
            "unexplored",
            "lack",
            "fails",
            "fail",
            "challenge",
            "however",
            "despite",
            "existing methods",
            "previous methods",
            "not clear",
            "empirical guidance",
            "generalization",
            "compute budget",
            "small datasets",
            "overfitting",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
            "auc",
            "table",
            "implementation",
            "formula",
        ],
        preferred_page_numbers=[1, 2, 3],
    ),
    "main_contribution": FieldRetrievalConfig(
        field_name="main_contribution",
        default_query=(
            "What are the main contributions, novelty, and proposed ideas of this paper?"
        ),
        preferred_section_keywords=[
            "abstract",
            "introduction",
            "method",
            "design",
            "architecture",
            "contribution",
        ],
        positive_keywords=[
            "contribution",
            "propose",
            "proposed",
            "introduce",
            "present",
            "novel",
            "framework",
            "architecture",
            "model",
            "hybrid",
            "lightweight",
            "mamba",
            "cnn",
            "patch embedding",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[1, 2, 3, 4],
    ),
    "method": FieldRetrievalConfig(
        field_name="method",
        default_query=(
            "What method, model, architecture, algorithm, or framework does this paper propose?"
        ),
        preferred_section_keywords=[
            "method",
            "methodology",
            "design",
            "architecture",
            "implementation",
            "selected design",
            "model",
        ],
        positive_keywords=[
            "method",
            "architecture",
            "framework",
            "model",
            "design",
            "implementation",
            "module",
            "block",
            "encoder",
            "classifier",
            "token",
            "patch",
            "mamba",
            "cnn",
            "fusion",
            "pooling",
            "regularization",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
    "datasets": FieldRetrievalConfig(
        field_name="datasets",
        default_query=(
            "What datasets, benchmarks, or data sources are used in this paper?"
        ),
        preferred_section_keywords=[
            "dataset",
            "datasets",
            "data",
            "experiments",
            "experimental",
        ],
        positive_keywords=[
            "dataset",
            "datasets",
            "benchmark",
            "data",
            "training",
            "testing",
            "validation",
            "medmnist",
            "kvasir",
            "pneumonia",
            "x-ray",
            "octmnist",
            "organ",
            "dermamnist",
            "bloodmnist",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
    "metrics": FieldRetrievalConfig(
        field_name="metrics",
        default_query=(
            "What evaluation metrics are used in this paper?"
        ),
        preferred_section_keywords=[
            "result",
            "analysis",
            "evaluation",
            "experiment",
            "auc",
            "statistical",
        ],
        positive_keywords=[
            "accuracy",
            "auc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "sensitivity",
            "gflops",
            "parameters",
            "oa",
            "metric",
            "metrics",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
    "baselines": FieldRetrievalConfig(
        field_name="baselines",
        default_query=(
            "What baseline methods or comparison models are used in this paper?"
        ),
        preferred_section_keywords=[
            "comparison",
            "relationships",
            "experiments",
            "result",
            "baseline",
        ],
        positive_keywords=[
            "baseline",
            "comparison",
            "compared",
            "models",
            "efficientnet",
            "deit",
            "medvit",
            "medmamba",
            "transformer",
            "cnn",
            "resnet",
            "vit",
            "state-of-the-art",
            "sota",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
    "results": FieldRetrievalConfig(
        field_name="results",
        default_query=(
            "What are the main experimental results, findings, and performance outcomes?"
        ),
        preferred_section_keywords=[
            "result",
            "analysis",
            "ablation",
            "comparison",
            "statistical",
            "cross-dataset",
        ],
        positive_keywords=[
            "result",
            "results",
            "performance",
            "outperform",
            "achieves",
            "achieved",
            "accuracy",
            "auc",
            "f1",
            "gflops",
            "parameters",
            "pareto",
            "trade-off",
            "evaluation",
            "ablation",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
    "limitations": FieldRetrievalConfig(
        field_name="limitations",
        default_query=(
            "What limitations, weaknesses, constraints, or failure cases are discussed?"
        ),
        preferred_section_keywords=[
            "discussion",
            "conclusion",
            "future work",
            "limitations",
            "result",
            "analysis",
        ],
        positive_keywords=[
            "limitation",
            "limitations",
            "weakness",
            "constraint",
            "failure",
            "fails",
            "challenge",
            "future",
            "further",
            "however",
            "although",
            "remaining",
            "limited",
            "trade-off",
            "small datasets",
            "generalization",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
    "future_work": FieldRetrievalConfig(
        field_name="future_work",
        default_query=(
            "What future work or next research directions are discussed?"
        ),
        preferred_section_keywords=[
            "conclusion",
            "future work",
            "discussion",
        ],
        positive_keywords=[
            "future",
            "further",
            "extend",
            "extension",
            "next",
            "remaining",
            "could",
            "should",
            "will",
            "direction",
            "research direction",
            "future work",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
            "auc",
            "table",
        ],
        preferred_page_numbers=[],
    ),
    "reproducibility": FieldRetrievalConfig(
        field_name="reproducibility",
        default_query=(
            "What implementation details, hyperparameters, optimizer, training settings, "
            "code availability, or reproducibility information are provided?"
        ),
        preferred_section_keywords=[
            "implementation",
            "selected design",
            "experiments",
            "dataset",
            "training",
        ],
        positive_keywords=[
            "implementation",
            "optimizer",
            "adamw",
            "learning rate",
            "batch",
            "epoch",
            "dropout",
            "dropblock",
            "cosine",
            "hyperparameter",
            "training",
            "loss",
            "focal",
            "label smoothing",
            "code",
            "available",
        ],
        negative_section_keywords=[
            "references",
            "bibliography",
        ],
        preferred_page_numbers=[],
    ),
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_preview(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def safe_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


class EvidenceBuilder:
    """
    Shared evidence builder for PaperMind v2.

    It improves retrieval quality by combining:
        - dense vector retrieval from Chroma
        - keyword scoring
        - section preference
        - page preference
        - noise penalties
        - compact context selection

    This module does not call the LLM.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        self.vector_store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

        self.retriever = PaperRetriever(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

    def build_evidence_pack(
        self,
        file_name: str,
        field_name: str,
        query: Optional[str] = None,
        candidate_top_k: int = 20,
        max_evidence_chunks: int = 4,
        max_chars_per_chunk: int = 1200,
    ) -> EvidencePack:
        if field_name not in FIELD_CONFIGS:
            raise ValueError(
                f"Unknown field_name='{field_name}'. "
                f"Available fields: {list(FIELD_CONFIGS.keys())}"
            )

        config = FIELD_CONFIGS[field_name]
        final_query = query or config.default_query

        all_chunks = self._load_chunks_for_file(file_name=file_name)

        if not all_chunks:
            raise RuntimeError(
                f"No chunks found for file_name='{file_name}'. "
                "Check the exact indexed file name or rebuild the vector store."
            )

        dense_scores = self._get_dense_scores(
            query=final_query,
            file_name=file_name,
            candidate_top_k=candidate_top_k,
        )

        scored_chunks = []

        for chunk in all_chunks:
            scored = self._score_chunk(
                chunk=chunk,
                config=config,
                dense_scores=dense_scores,
                max_chars_per_chunk=max_chars_per_chunk,
            )

            scored_chunks.append(scored)

        selected_chunks = self._select_diverse_chunks(
            scored_chunks=scored_chunks,
            max_evidence_chunks=max_evidence_chunks,
        )

        ranked_chunks = []

        for rank, chunk in enumerate(selected_chunks, start=1):
            ranked_chunks.append(
                EvidenceChunk(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    file_name=chunk.file_name,
                    paper_title=chunk.paper_title,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    source=chunk.source,
                    text=chunk.text,
                    text_preview=chunk.text_preview,
                    dense_score=chunk.dense_score,
                    keyword_score=chunk.keyword_score,
                    section_score=chunk.section_score,
                    page_score=chunk.page_score,
                    noise_penalty=chunk.noise_penalty,
                    final_score=chunk.final_score,
                )
            )

        return EvidencePack(
            file_name=file_name,
            field_name=field_name,
            query=final_query,
            evidence_chunks=ranked_chunks,
        )

    def build_multiple_evidence_packs(
        self,
        file_name: str,
        field_names: list[str],
        candidate_top_k: int = 20,
        max_evidence_chunks: int = 4,
        max_chars_per_chunk: int = 1200,
    ) -> dict[str, EvidencePack]:
        packs = {}

        for field_name in field_names:
            packs[field_name] = self.build_evidence_pack(
                file_name=file_name,
                field_name=field_name,
                candidate_top_k=candidate_top_k,
                max_evidence_chunks=max_evidence_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
            )

        return packs

    def _load_chunks_for_file(self, file_name: str) -> list[dict]:
        count = self.vector_store.count()

        if count == 0:
            return []

        results = self.vector_store.collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        chunks = []

        for index, metadata in enumerate(metadatas):
            if metadata.get("file_name") != file_name:
                continue

            text = documents[index] if index < len(documents) else ""
            chunk_id = ids[index] if index < len(ids) else f"unknown_{index}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                }
            )

        return chunks

    def _get_dense_scores(
        self,
        query: str,
        file_name: str,
        candidate_top_k: int,
    ) -> dict[str, dict]:
        dense_scores = {}

        try:
            retrieved_chunks = self.retriever.search(
                query=query,
                top_k=candidate_top_k,
                file_name=file_name,
            )

            for chunk in retrieved_chunks:
                dense_scores[chunk.chunk_id] = {
                    "rank": chunk.rank,
                    "relevance_score": float(chunk.relevance_score),
                    "distance": float(chunk.distance),
                }

        except Exception as error:
            logger.warning("Dense retrieval failed: %s", error)

        return dense_scores

    def _score_chunk(
        self,
        chunk: dict,
        config: FieldRetrievalConfig,
        dense_scores: dict[str, dict],
        max_chars_per_chunk: int,
    ) -> EvidenceChunk:
        chunk_id = str(chunk["chunk_id"])
        text = str(chunk.get("text", ""))
        metadata = chunk.get("metadata", {})

        file_name = str(metadata.get("file_name", "Unknown"))
        paper_title = str(metadata.get("paper_title", "Unknown"))
        page_number = safe_int(metadata.get("page_number", -1))
        section_title = str(metadata.get("section_title", "Unknown"))
        source = str(metadata.get("source", "Unknown source"))

        dense_info = dense_scores.get(chunk_id, {})
        dense_score = float(dense_info.get("relevance_score", 0.0))

        keyword_score = self._keyword_score(
            text=text,
            keywords=config.positive_keywords,
        )

        section_score = self._section_score(
            section_title=section_title,
            text=text,
            preferred_section_keywords=config.preferred_section_keywords,
        )

        page_score = self._page_score(
            page_number=page_number,
            preferred_page_numbers=config.preferred_page_numbers,
            section_title=section_title,
            text=text,
            field_name=config.field_name,
        )

        noise_penalty = self._noise_penalty(
            text=text,
            section_title=section_title,
            negative_section_keywords=config.negative_section_keywords,
            field_name=config.field_name,
        )

        final_score = (
            (0.30 * dense_score)
            + (0.30 * keyword_score)
            + (0.25 * section_score)
            + (0.15 * page_score)
            - noise_penalty
        )

        return EvidenceChunk(
            rank=0,
            chunk_id=chunk_id,
            file_name=file_name,
            paper_title=paper_title,
            page_number=page_number,
            section_title=section_title,
            source=source,
            text=text,
            text_preview=clean_preview(text, max_chars=max_chars_per_chunk),
            dense_score=dense_score,
            keyword_score=keyword_score,
            section_score=section_score,
            page_score=page_score,
            noise_penalty=noise_penalty,
            final_score=final_score,
        )

    @staticmethod
    def _keyword_score(text: str, keywords: list[str]) -> float:
        if not text or not keywords:
            return 0.0

        normalized = normalize_text(text)

        score = 0.0

        for keyword in keywords:
            keyword_norm = normalize_text(keyword)

            if not keyword_norm:
                continue

            if " " in keyword_norm:
                if keyword_norm in normalized:
                    score += 2.0
            else:
                count = len(
                    re.findall(
                        rf"\b{re.escape(keyword_norm)}\b",
                        normalized,
                    )
                )
                score += min(count, 3) * 0.6

        return min(score / 8.0, 1.0)

    @staticmethod
    def _section_score(
        section_title: str,
        text: str,
        preferred_section_keywords: list[str],
    ) -> float:
        section_norm = normalize_text(section_title)
        text_start = normalize_text(text[:700])

        if not preferred_section_keywords:
            return 0.0

        score = 0.0

        for keyword in preferred_section_keywords:
            keyword_norm = normalize_text(keyword)

            if keyword_norm in section_norm:
                score = max(score, 1.0)

            elif keyword_norm in text_start:
                score = max(score, 0.65)

        return score

    @staticmethod
    def _page_score(
        page_number: int,
        preferred_page_numbers: list[int],
        section_title: str,
        text: str,
        field_name: str,
    ) -> float:
        section_norm = normalize_text(section_title)
        text_start = normalize_text(text[:1000])

        if page_number in preferred_page_numbers:
            return 1.0

        if field_name in {"motivation", "problem", "research_gap", "main_contribution"}:
            if page_number in {1, 2}:
                return 0.8

            if "abstract" in text_start or "introduction" in section_norm:
                return 0.8

        if field_name in {"limitations", "future_work"}:
            if "conclusion" in section_norm or "future" in section_norm:
                return 1.0

            if "future work" in text_start or "conclusion" in text_start:
                return 0.8

        return 0.0

    @staticmethod
    def _noise_penalty(
        text: str,
        section_title: str,
        negative_section_keywords: list[str],
        field_name: str,
    ) -> float:
        section_norm = normalize_text(section_title)
        text_norm = normalize_text(text)

        penalty = 0.0

        for keyword in negative_section_keywords:
            if normalize_text(keyword) in section_norm:
                penalty += 0.25

        citation_count = len(re.findall(r"\[\d+\]", text))
        url_count = text_norm.count("http") + text_norm.count("arxiv")

        if field_name not in {"baselines", "literature_review"}:
            if citation_count >= 8:
                penalty += 0.35
            elif citation_count >= 5:
                penalty += 0.20

            if url_count >= 2:
                penalty += 0.25

        alphabetic_chars = len(re.findall(r"[a-zA-Z]", text))
        symbol_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))

        if alphabetic_chars > 0:
            symbol_ratio = symbol_chars / alphabetic_chars
            if symbol_ratio > 0.45:
                penalty += 0.25

        table_like_patterns = [
            "auc",
            "spec.",
            "sens.",
            "gflops",
            "parameters",
            "table",
        ]

        if field_name not in {"metrics", "results", "baselines"}:
            table_hits = sum(1 for pattern in table_like_patterns if pattern in text_norm)
            if table_hits >= 3:
                penalty += 0.20

        if len(text.split()) < 35:
            penalty += 0.25

        return min(penalty, 0.85)

    @staticmethod
    def _select_diverse_chunks(
        scored_chunks: list[EvidenceChunk],
        max_evidence_chunks: int,
    ) -> list[EvidenceChunk]:
        """
        Select strong and diverse evidence chunks.

        Important:
        We do not want to fill max_evidence_chunks with weak chunks.
        For local LLMs, fewer high-quality chunks are better than many noisy chunks.
        """
        sorted_chunks = sorted(
            scored_chunks,
            key=lambda chunk: chunk.final_score,
            reverse=True,
        )

        selected = []
        page_counts: dict[int, int] = {}
        section_counts: dict[str, int] = {}

        for chunk in sorted_chunks:
            if len(selected) >= max_evidence_chunks:
                break

            if not EvidenceBuilder._passes_quality_gate(chunk):
                continue

            page_count = page_counts.get(chunk.page_number, 0)
            section_count = section_counts.get(chunk.section_title, 0)

            if page_count >= 2:
                continue

            if section_count >= 2:
                continue

            selected.append(chunk)

            page_counts[chunk.page_number] = page_count + 1
            section_counts[chunk.section_title] = section_count + 1

        return selected

    @staticmethod
    def _passes_quality_gate(chunk: EvidenceChunk) -> bool:
        """
        Prevent weak chunks from being selected.

        A chunk should either:
        - have a strong final score, or
        - have meaningful keyword/section support.

        This avoids irrelevant table/reference chunks being selected only because
        dense retrieval or page score was mildly positive.
        """
        if chunk.final_score >= 0.30:
            return True

        if chunk.final_score >= 0.22 and (
            chunk.keyword_score >= 0.25 or chunk.section_score >= 0.65
        ):
            return True

        return False


def evidence_pack_to_markdown(pack: EvidencePack) -> str:
    lines = [
        "# Evidence Pack",
        "",
        f"**File:** `{pack.file_name}`",
        f"**Field:** `{pack.field_name}`",
        f"**Query:** {pack.query}",
        "",
    ]

    for chunk in pack.evidence_chunks:
        lines.extend(
            [
                f"## Evidence {chunk.rank}",
                "",
                f"- **Source:** {chunk.source}",
                f"- **Page:** {chunk.page_number}",
                f"- **Section:** {chunk.section_title}",
                f"- **Final Score:** {chunk.final_score:.4f}",
                f"- **Dense Score:** {chunk.dense_score:.4f}",
                f"- **Keyword Score:** {chunk.keyword_score:.4f}",
                f"- **Section Score:** {chunk.section_score:.4f}",
                f"- **Page Score:** {chunk.page_score:.4f}",
                f"- **Noise Penalty:** {chunk.noise_penalty:.4f}",
                "",
                "```text",
                chunk.text_preview,
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def print_evidence_pack(pack: EvidencePack):
    print("\n")
    print("=" * 120)
    print("EVIDENCE PACK")
    print("=" * 120)
    print(f"File: {pack.file_name}")
    print(f"Field: {pack.field_name}")
    print(f"Query: {pack.query}")

    for chunk in pack.evidence_chunks:
        print("\n" + "-" * 120)
        print(f"Evidence Rank: {chunk.rank}")
        print(f"Source: {chunk.source}")
        print(f"Page: {chunk.page_number}")
        print(f"Section: {chunk.section_title}")
        print(f"Final Score: {chunk.final_score:.4f}")
        print(
            "Components: "
            f"dense={chunk.dense_score:.4f}, "
            f"keyword={chunk.keyword_score:.4f}, "
            f"section={chunk.section_score:.4f}, "
            f"page={chunk.page_score:.4f}, "
            f"penalty={chunk.noise_penalty:.4f}"
        )
        print("-" * 120)
        print(chunk.text_preview)

    print("-" * 120)


def save_pack_outputs(
    pack: EvidencePack,
    output_json: Optional[str] = None,
    output_md: Optional[str] = None,
):
    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(pack.to_dict(), file, indent=4, ensure_ascii=False)

        print(f"\nSaved evidence JSON to: {output_json_path}")

    if output_md:
        output_md_path = Path(output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_md_path, "w", encoding="utf-8") as file:
            file.write(evidence_pack_to_markdown(pack))

        print(f"Saved evidence Markdown to: {output_md_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build field-specific evidence packs for PaperMind-RAG."
    )

    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List available evidence fields and exit.",
    )

    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Exact indexed PDF file name.",
    )

    parser.add_argument(
        "--field",
        type=str,
        default=None,
        choices=list(FIELD_CONFIGS.keys()),
        help="Evidence field to build.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional custom query. If omitted, field default query is used.",
    )

    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=20,
        help="How many dense retrieval candidates to inspect internally.",
    )

    parser.add_argument(
        "--max-evidence",
        type=int,
        default=4,
        help="How many compact evidence chunks to return.",
    )

    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=1200,
        help="Maximum characters per evidence chunk preview.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evidence pack JSON.",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional path to save evidence pack Markdown.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_fields:
        print("\nAvailable fields:")
        for field_name in FIELD_CONFIGS:
            print(f"- {field_name}")
        return

    if not args.file_name:
        raise ValueError("Please provide --file-name.")

    if not args.field:
        raise ValueError("Please provide --field.")

    builder = EvidenceBuilder()

    pack = builder.build_evidence_pack(
        file_name=args.file_name,
        field_name=args.field,
        query=args.query,
        candidate_top_k=args.candidate_top_k,
        max_evidence_chunks=args.max_evidence,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )

    print_evidence_pack(pack)

    save_pack_outputs(
        pack=pack,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()