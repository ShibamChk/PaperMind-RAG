from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
from typing import Optional

import fitz


KNOWN_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "proposed method",
    "experimental setup",
    "experiments",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "references",
    "appendix",
}


@dataclass
class ParsedSection:
    document_id: str
    paper_title: str
    file_name: str
    file_path: str
    page_number: int
    section_title: str
    text: str


@dataclass
class ParsedPaper:
    document_id: str
    paper_title: str
    file_name: str
    file_path: str
    total_pages: int
    sections: list[ParsedSection]


class PDFParser:
    """
    Section-aware PDF parser for research papers.

    This parser uses PyMuPDF to extract page text and a heuristic method
    to detect section headings such as Abstract, Introduction, Methodology,
    Results, Limitations, and Conclusion.

    The goal is not perfect scientific PDF parsing. The goal is to create
    useful metadata for downstream RAG retrieval.
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {self.pdf_path}")

    def parse(self) -> ParsedPaper:
        document = fitz.open(self.pdf_path)

        document_id = self._create_document_id(self.pdf_path)
        paper_title = self._extract_title(document)

        sections: list[ParsedSection] = []

        current_section_title = "Unknown"
        current_text_lines: list[str] = []
        current_page_number = 1

        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1
            page_text = page.get_text("text")

            cleaned_lines = self._clean_page_text(page_text)

            for line in cleaned_lines:
                if self._is_probable_section_heading(line):
                    if current_text_lines:
                        section_text = "\n".join(current_text_lines).strip()

                        if section_text:
                            sections.append(
                                ParsedSection(
                                    document_id=document_id,
                                    paper_title=paper_title,
                                    file_name=self.pdf_path.name,
                                    file_path=str(self.pdf_path),
                                    page_number=current_page_number,
                                    section_title=current_section_title,
                                    text=section_text,
                                )
                            )

                    current_section_title = self._normalize_section_title(line)
                    current_text_lines = []
                    current_page_number = page_number
                else:
                    current_text_lines.append(line)

            # If a page has no detected section change, keep accumulating text
            # under the current section.

        if current_text_lines:
            section_text = "\n".join(current_text_lines).strip()

            if section_text:
                sections.append(
                    ParsedSection(
                        document_id=document_id,
                        paper_title=paper_title,
                        file_name=self.pdf_path.name,
                        file_path=str(self.pdf_path),
                        page_number=current_page_number,
                        section_title=current_section_title,
                        text=section_text,
                    )
                )
        total_pages = len(document)

        document.close()

        return ParsedPaper(
            document_id=document_id,
            paper_title=paper_title,
            file_name=self.pdf_path.name,
            file_path=str(self.pdf_path),
            total_pages=total_pages,
            sections=sections,
        )

    @staticmethod
    def to_dict(parsed_paper: ParsedPaper) -> dict:
        return {
            "document_id": parsed_paper.document_id,
            "paper_title": parsed_paper.paper_title,
            "file_name": parsed_paper.file_name,
            "file_path": parsed_paper.file_path,
            "total_pages": parsed_paper.total_pages,
            "sections": [asdict(section) for section in parsed_paper.sections],
        }

    def _extract_title(self, document: fitz.Document) -> str:
        metadata_title = document.metadata.get("title", "")

        if metadata_title and metadata_title.strip():
            return self._clean_whitespace(metadata_title)

        if len(document) == 0:
            return self.pdf_path.stem

        first_page_text = document[0].get_text("text")
        lines = self._clean_page_text(first_page_text)

        candidate_lines = []

        for line in lines[:15]:
            if len(line.split()) >= 3 and not self._looks_like_author_line(line):
                candidate_lines.append(line)

        if candidate_lines:
            return self._clean_whitespace(candidate_lines[0])

        return self.pdf_path.stem

    @staticmethod
    def _create_document_id(pdf_path: Path) -> str:
        file_stat = pdf_path.stat()
        raw = f"{pdf_path.name}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _clean_page_text(self, text: str) -> list[str]:
        lines = []

        for raw_line in text.splitlines():
            line = self._clean_whitespace(raw_line)

            if not line:
                continue

            if self._is_noise_line(line):
                continue

            lines.append(line)

        return lines

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        lower_line = line.lower()

        if len(line) <= 1:
            return True

        if lower_line.startswith("arxiv:"):
            return True

        if lower_line.startswith("copyright"):
            return True

        if re.fullmatch(r"\d+", line):
            return True

        return False

    @staticmethod
    def _looks_like_author_line(line: str) -> bool:
        lowered = line.lower()

        author_indicators = [
            "university",
            "department",
            "institute",
            "@",
            "school of",
            "college",
        ]

        return any(indicator in lowered for indicator in author_indicators)

    def _is_probable_section_heading(self, line: str) -> bool:
        normalized = self._normalize_section_title(line)
        normalized_lower = normalized.lower()

        if normalized_lower in KNOWN_SECTION_HEADINGS:
            return True

        # Examples:
        # 1 Introduction
        # 2. Related Work
        # 3.1 Model Architecture
        numbered_heading_pattern = r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9,\-\s:]{2,80}$"

        if re.match(numbered_heading_pattern, line):
            return True

        # Short all-caps headings
        if line.isupper() and 3 <= len(line) <= 80:
            return True

        return False

    @staticmethod
    def _normalize_section_title(line: str) -> str:
        line = re.sub(r"^\d+(\.\d+)*\.?\s*", "", line)
        line = line.strip(" .:-")
        line = re.sub(r"\s+", " ", line)

        if not line:
            return "Unknown"

        return line