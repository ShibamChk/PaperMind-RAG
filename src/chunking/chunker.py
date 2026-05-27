from __future__ import annotations

from pathlib import Path
import sys
import json
import argparse
import hashlib
import re
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import (
    DEFAULT_PARSED_OUTPUT_PATH,
    DEFAULT_CHUNKED_OUTPUT_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_WORDS,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_jsonl(input_path: str | Path) -> list[dict]:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    records = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {input_path}"
                ) from error

    return records


def save_jsonl(records: Iterable[dict], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Saved %s chunks to %s", count, output_path)


def clean_text_for_chunking(text: str) -> str:
    """
    Clean section text before chunking.

    We keep academic content but remove excessive whitespace.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def split_text_into_words(text: str) -> list[str]:
    return text.split()


def create_chunk_id(
    document_id: str,
    section_title: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    raw = f"{document_id}_{section_title}_{page_number}_{chunk_index}_{text[:100]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def chunk_words(
    words: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split a list of words into overlapping chunks.

    Example:
        chunk_size = 800
        chunk_overlap = 150

    Chunk 1: words 0 to 799
    Chunk 2: words 650 to 1449
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if chunk:
            chunks.append(" ".join(chunk))

        if end >= len(words):
            break

        start = end - chunk_overlap

    return chunks


def make_source_string(record: dict) -> str:
    file_name = record.get("file_name", "unknown_file")
    page_number = record.get("page_number", "unknown_page")
    section_title = record.get("section_title", "Unknown")

    return f"{file_name} | page {page_number} | {section_title}"


def chunk_section_record(
    record: dict,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int,
) -> list[dict]:
    """
    Convert one parsed section record into one or more RAG chunks.
    """
    required_fields = [
        "document_id",
        "paper_title",
        "file_name",
        "file_path",
        "page_number",
        "section_title",
        "text",
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing required field in parsed record: {field}")

    text = clean_text_for_chunking(record["text"])

    if not text:
        return []

    words = split_text_into_words(text)

    if len(words) < min_chunk_words:
        return []

    raw_chunks = chunk_words(
        words=words,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for chunk_index, chunk_text in enumerate(raw_chunks):
        chunk_word_count = len(chunk_text.split())

        if chunk_word_count < min_chunk_words:
            continue

        chunk_id = create_chunk_id(
            document_id=record["document_id"],
            section_title=record["section_title"],
            page_number=int(record["page_number"]),
            chunk_index=chunk_index,
            text=chunk_text,
        )

        chunk = {
            "chunk_id": chunk_id,
            "document_id": record["document_id"],
            "paper_title": record["paper_title"],
            "file_name": record["file_name"],
            "file_path": record["file_path"],
            "page_number": int(record["page_number"]),
            "section_title": record["section_title"],
            "chunk_index": chunk_index,
            "word_count": chunk_word_count,
            "text": chunk_text,
            "source": make_source_string(record),
        }

        chunks.append(chunk)

    return chunks


def chunk_parsed_sections(
    parsed_records: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int,
) -> list[dict]:
    all_chunks = []

    for record in parsed_records:
        section_chunks = chunk_section_record(
            record=record,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_words=min_chunk_words,
        )

        all_chunks.extend(section_chunks)

    return all_chunks


def summarize_chunks(chunks: list[dict]):
    if not chunks:
        logger.warning("No chunks were created.")
        return

    logger.info("Total chunks created: %s", len(chunks))

    unique_documents = sorted({chunk["document_id"] for chunk in chunks})
    logger.info("Unique documents: %s", len(unique_documents))

    section_counts = {}

    for chunk in chunks:
        section_title = chunk["section_title"]
        section_counts[section_title] = section_counts.get(section_title, 0) + 1

    logger.info("Top section counts:")

    for section_title, count in sorted(
        section_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        logger.info("  %s: %s", section_title, count)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create RAG chunks from parsed research paper sections."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_PARSED_OUTPUT_PATH),
        help="Path to parsed papers JSONL.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_CHUNKED_OUTPUT_PATH),
        help="Path to output chunks JSONL.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size in words.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Chunk overlap in words.",
    )

    parser.add_argument(
        "--min-chunk-words",
        type=int,
        default=MIN_CHUNK_WORDS,
        help="Minimum number of words required to keep a chunk.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting chunking")
    logger.info("Input path: %s", args.input_path)
    logger.info("Output path: %s", args.output_path)
    logger.info("Chunk size: %s words", args.chunk_size)
    logger.info("Chunk overlap: %s words", args.chunk_overlap)

    parsed_records = load_jsonl(args.input_path)

    logger.info("Loaded %s parsed section records", len(parsed_records))

    chunks = chunk_parsed_sections(
        parsed_records=parsed_records,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_words=args.min_chunk_words,
    )

    summarize_chunks(chunks)

    save_jsonl(
        records=chunks,
        output_path=args.output_path,
    )

    logger.info("Chunking complete")


if __name__ == "__main__":
    main()