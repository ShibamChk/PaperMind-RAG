from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_PARSED_OUTPUT_PATH = PROCESSED_DATA_DIR / "parsed_papers.jsonl"
DEFAULT_CHUNKED_OUTPUT_PATH = PROCESSED_DATA_DIR / "paper_chunks.jsonl"

CHROMA_DB_DIR = VECTOR_STORE_DIR / "chroma"
CHROMA_COLLECTION_NAME = "papermind_research_chunks"

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
}

# Chunking defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
MIN_CHUNK_WORDS = 40

# Embedding defaults
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32