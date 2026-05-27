from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import (
    DOCS_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_PARSED_OUTPUT_PATH,
    SUPPORTED_DOCUMENT_EXTENSIONS,
)
from src.parsing.pdf_parser import PDFParser
from src.utils.logger import get_logger


logger = get_logger(__name__)


def find_supported_documents(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = []

    for extension in SUPPORTED_DOCUMENT_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{extension}"))

    files = sorted(files)

    return files


def save_sections_as_jsonl(parsed_papers: list[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_sections = 0

    with open(output_path, "w", encoding="utf-8") as file:
        for paper in parsed_papers:
            for section in paper["sections"]:
                file.write(json.dumps(section, ensure_ascii=False) + "\n")
                total_sections += 1

    logger.info("Saved %s parsed sections to %s", total_sections, output_path)


def load_and_parse_documents(
    input_dir: str | Path = DOCS_DIR,
    output_path: str | Path = DEFAULT_PARSED_OUTPUT_PATH,
) -> list[dict]:
    documents = find_supported_documents(input_dir)

    if not documents:
        raise FileNotFoundError(
            f"No supported documents found in {input_dir}. "
            f"Supported extensions: {SUPPORTED_DOCUMENT_EXTENSIONS}"
        )

    logger.info("Found %s document(s) in %s", len(documents), input_dir)

    parsed_papers = []

    for document_path in tqdm(documents, desc="Parsing documents"):
        try:
            parser = PDFParser(document_path)
            parsed_paper = parser.parse()
            parsed_paper_dict = PDFParser.to_dict(parsed_paper)

            parsed_papers.append(parsed_paper_dict)

            logger.info(
                "Parsed %s | title=%s | sections=%s",
                document_path.name,
                parsed_paper.paper_title,
                len(parsed_paper.sections),
            )

        except Exception as error:
            logger.exception("Failed to parse %s: %s", document_path, error)

    if not parsed_papers:
        raise RuntimeError("No documents were parsed successfully.")

    save_sections_as_jsonl(
        parsed_papers=parsed_papers,
        output_path=output_path,
    )

    return parsed_papers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and parse research paper PDFs into JSONL sections."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DOCS_DIR),
        help="Directory containing PDF files.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_PARSED_OUTPUT_PATH),
        help="Output JSONL path.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting document ingestion")
    logger.info("Input directory: %s", args.input_dir)
    logger.info("Output path: %s", args.output_path)

    load_and_parse_documents(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )

    logger.info("Document ingestion complete")


if __name__ == "__main__":
    main()