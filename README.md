# PaperMind-RAG

PaperMind-RAG is a local-first research paper intelligence system built with Retrieval-Augmented Generation. It helps users analyze academic papers, ask citation-grounded questions, generate structured paper summaries, compare multiple papers, identify research gaps, and create reviewer-style critiques.

Unlike a basic PDF chatbot, PaperMind-RAG is designed as a research workflow assistant. It combines PDF parsing, section-aware chunking, embedding-based retrieval, ChromaDB vector storage, local LLM generation through Ollama, and a Streamlit interface.

---

## Project Overview

Reading research papers is time-consuming because important information is often spread across the abstract, introduction, related work, methodology, experiments, discussion, and conclusion sections. PaperMind-RAG was built to make this process faster, more structured, and more useful for research workflows.

The system can:

- Upload and index research paper PDFs
- Ask citation-grounded questions over indexed papers
- Generate structured Paper Cards
- Compare multiple research papers
- Identify research gaps and possible future directions
- Generate conference-style reviewer reports
- Run locally using Ollama without requiring paid API access
- Provide an interactive Streamlit interface

---

## Motivation

Most RAG projects are simple document chatbots. PaperMind-RAG focuses on a more specific and practical research workflow.

Researchers and students often need to answer questions such as:

- What problem does this paper solve?
- What is the main contribution?
- What method or architecture is proposed?
- What datasets and evaluation metrics are used?
- What are the limitations?
- How does this paper compare with related work?
- What research gaps can be explored next?
- What would a reviewer say about this paper?

PaperMind-RAG addresses these tasks using a modular RAG pipeline and task-specific generation modules.

---

## Key Features

### Citation-Grounded Question Answering

Ask natural language questions over uploaded research papers. The system retrieves relevant chunks from the indexed papers and generates an answer using the retrieved context.

Example questions:

```text
What is the main contribution of this paper?
How does EvolveGCN use RNNs to evolve GCN parameters?
What datasets were used in this study?
What are the limitations of the proposed method?
```

---

### Paper Card Generator

Generates a structured research paper profile containing:

- Problem
- Motivation
- Main contribution
- Proposed method
- Model architecture
- Datasets
- Tasks
- Evaluation metrics
- Baselines
- Key results
- Limitations
- Future work
- Reproducibility notes

This feature is useful for literature review preparation and quick paper understanding.

---

### Multi-Paper Comparison Matrix

Compares multiple papers across research dimensions such as:

- Problem statement
- Main contribution
- Method
- Architecture or framework
- Datasets
- Tasks
- Evaluation metrics
- Baselines
- Key results
- Limitations
- Unique strengths

The comparison module generates one structured row per paper and then synthesizes similarities, differences, and research gaps.

---

### Research Gap Finder

Analyzes one or more papers and generates a research gap report containing:

- Common limitations
- Dataset gaps
- Evaluation gaps
- Scalability gaps
- Robustness gaps
- Reproducibility gaps
- Proposed research directions
- Suggested experiments
- Suggested ablation studies

This feature is designed to support thesis planning, research ideation, and literature review analysis.

---

### Reviewer Mode

Generates a conference-style review for a selected paper, including:

- Review summary
- Main contributions
- Strengths
- Weaknesses
- Novelty assessment
- Technical soundness
- Experimental quality
- Missing experiments
- Reproducibility concerns
- Questions for authors
- Final recommendation
- Reviewer confidence

This mode simulates a structured academic review process.

---

### Local LLM Support with Ollama

PaperMind-RAG supports local generation through Ollama. This allows the system to run without paid API access.

Supported provider options:

```text
ollama
openai
```

Recommended local model:

```text
qwen3:14b
```

The system also supports switching models from the Streamlit interface.

---

## System Architecture

```text
PDF Research Papers
        ↓
PDF Parsing
        ↓
Section-Aware Text Extraction
        ↓
Chunking
        ↓
Embedding Generation
        ↓
ChromaDB Vector Store
        ↓
Citation-Ready Retrieval
        ↓
Local or Cloud LLM Generation
        ↓
Research Outputs
```

---

## RAG Pipeline

PaperMind-RAG follows a Retrieval-Augmented Generation workflow.

### 1. Document Ingestion

Research papers are uploaded into the local `docs/` directory.

### 2. PDF Parsing

The parser extracts text, page information, section metadata, and document identifiers from each PDF.

### 3. Chunking

Extracted paper text is split into manageable chunks while preserving metadata such as:

- Paper title
- File name
- Page number
- Section title
- Source citation

### 4. Embedding Generation

Each chunk is converted into a dense vector representation using SentenceTransformers.

Default embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 5. Vector Storage

Embeddings and metadata are stored in ChromaDB.

### 6. Retrieval

For each query or task, the retriever searches the vector database and returns the most relevant chunks.

### 7. Generation

The selected context is passed to an LLM through Ollama or OpenAI. The model generates structured, source-aware outputs.

---

## Project Structure

```text
papermind-rag/
├── app/
│   ├── streamlit_app.py
│   └── api.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
│
├── docs/
│   └── uploaded PDF papers
│
├── reports/
│   └── generated local outputs
│
├── src/
│   ├── chunking/
│   │   └── chunker.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── evaluation/
│   │   └── rag_evaluator.py
│   │
│   ├── generation/
│   │   ├── answer_generator.py
│   │   ├── paper_card_generator.py
│   │   ├── comparison_generator.py
│   │   ├── gap_finder.py
│   │   └── reviewer.py
│   │
│   ├── ingestion/
│   │   └── document_loader.py
│   │
│   ├── parsing/
│   │   └── pdf_parser.py
│   │
│   ├── pipelines/
│   │   └── rag_pipeline.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   └── utils/
│       └── logger.py
│
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── main.py
```

---

## Technology Stack

```text
Python
Streamlit
PyMuPDF
SentenceTransformers
ChromaDB
Ollama
OpenAI API support
Pandas
NumPy
Scikit-learn
Pydantic
FastAPI
Uvicorn
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ShibamChk/PaperMind-RAG.git
cd PaperMind-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

For Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root.

Example:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=qwen3:14b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=900

OPENAI_API_KEY=
OPENAI_MODEL_NAME=gpt-4o-mini

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_DIR=data/vector_store/chroma
DEFAULT_CHUNK_SIZE=800
DEFAULT_CHUNK_OVERLAP=150
```

The `.env` file should not be committed to GitHub.

---

## Ollama Setup

Install Ollama, then pull a local model:

```bash
ollama pull qwen3:14b
```

Check available models:

```bash
ollama list
```

Run Ollama if needed:

```bash
ollama serve
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

The app supports:

- Uploading PDFs
- Rebuilding the vector index
- Asking citation-grounded questions
- Generating Paper Cards
- Comparing papers
- Finding research gaps
- Running Reviewer Mode

---

## Command-Line Usage

### 1. Parse uploaded PDFs

Place PDFs inside:

```text
docs/
```

Then run:

```bash
python src/ingestion/document_loader.py --input-dir docs --output-path data/processed/parsed_papers.jsonl
```

### 2. Create chunks

```bash
python src/chunking/chunker.py --input-path data/processed/parsed_papers.jsonl --output-path data/processed/paper_chunks.jsonl
```

### 3. Build vector store

```bash
python src/retrieval/vector_store.py --chunks-path data/processed/paper_chunks.jsonl --persist-dir data/vector_store/chroma --reset
```

### 4. Test retrieval

```bash
python src/retrieval/retriever.py --query "What is the main contribution of this paper?" --top-k 3 --show-context
```

### 5. Ask a RAG question

```bash
python src/pipelines/rag_pipeline.py --question "What is the main contribution of EvolveGCN?" --top-k 3 --llm-provider ollama --llm-model-name qwen3:14b
```

### 6. Generate Paper Card

```bash
python src/generation/paper_card_generator.py --file-name evolvegc.pdf --top-k-per-query 2 --llm-provider ollama --llm-model-name qwen3:14b --output-json reports/paper_card.json --output-md reports/paper_card.md
```

### 7. Compare Papers

```bash
python src/generation/comparison_generator.py --file-names evolvegc.pdf tgn.pdf --top-k-per-query 2 --llm-provider ollama --llm-model-name qwen3:14b --output-json reports/comparison.json --output-md reports/comparison.md
```

### 8. Generate Research Gap Report

```bash
python src/generation/gap_finder.py --file-names evolvegc.pdf tgn.pdf --top-k-per-query 1 --llm-provider ollama --llm-model-name qwen3:14b --output-json reports/gap_report.json --output-md reports/gap_report.md
```

### 9. Generate Reviewer Report

```bash
python src/generation/reviewer.py --file-name evolvegc.pdf --top-k-per-query 1 --llm-provider ollama --llm-model-name qwen3:14b --output-json reports/review.json --output-md reports/review.md
```

---

## Example Workflow

```text
1. Add PDFs to docs/
2. Run the indexing pipeline
3. Ask citation-grounded questions
4. Generate Paper Cards
5. Compare related papers
6. Identify research gaps
7. Generate reviewer-style critiques
```

---

## Local Hardware Notes

PaperMind-RAG can run with a local LLM through Ollama. Larger local models require more memory and may be slower on consumer hardware.

For limited hardware, use:

```text
top_k = 1 or 2
```

Higher `top_k` values provide more retrieved context but increase memory usage and generation time.

Recommended setup for local usage:

```text
LLM: qwen3:14b or smaller
Top K: 1 to 2
Embedding model: all-MiniLM-L6-v2
Vector database: ChromaDB
```

---

## Why RAG Instead of Fine-Tuning?

RAG is better suited for this project because academic papers change frequently and require exact source grounding.

Fine-tuning would require retraining or updating model weights. RAG allows new papers to be added by simply parsing, chunking, embedding, and indexing them.

| Fine-Tuning | RAG |
|---|---|
| Changes model weights | Keeps model fixed |
| Expensive to update | Easy to update |
| Weak for exact citations | Stronger for citations |
| Best for learning behavior or style | Best for document-grounded QA |
| Harder to inspect | Easier to debug through retrieved sources |

---

## Engineering Highlights

This project demonstrates:

- End-to-end RAG pipeline design
- Local-first LLM integration with Ollama
- PDF parsing and metadata extraction
- Chunking and embedding generation
- ChromaDB vector search
- Citation-ready retrieval
- Modular generation pipelines
- Structured JSON generation and repair fallback
- Streamlit application development
- Professional project organization
- Git-based project versioning

---

## Current Limitations

PaperMind-RAG is designed as a research assistance system, not a replacement for human academic judgment.

Current limitations include:

- PDF parsing may be imperfect for complex layouts, equations, tables, and scanned documents.
- Retrieval is currently based mainly on semantic vector search.
- Citations are chunk-level rather than sentence-level.
- Local LLM output quality depends on the selected model and available hardware.
- Some implicit research fields may require improved section-aware retrieval and evidence packing.
- Generated results should always be verified against the original papers.

---

## Future Improvements

Planned improvements include:

- Retrieval debugger for inspecting retrieved chunks
- Improved section detection
- Paragraph-aware chunking
- Hybrid retrieval using vector search and keyword search
- Reranking for better evidence selection
- Field-specific evidence packs for Paper Card, Gap Finder, Comparison, and Reviewer Mode
- Paper-level summary memory
- Better table extraction from research papers
- RAG evaluation metrics
- FastAPI backend integration
- Docker support
- Cloud deployment

---

## Repository Data Policy

The repository does not include uploaded research papers, generated local reports, processed data, vector databases, or environment secrets.

Ignored local assets include:

```text
docs/*.pdf
data/processed/
data/vector_store/
reports/
.env
.venv/
```

This keeps the repository lightweight and safe to share.

---

## Author

Shibam Chakraborty