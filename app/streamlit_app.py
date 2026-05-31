from __future__ import annotations

from pathlib import Path
import sys
import json
import traceback

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import (
    DOCS_DIR,
    DEFAULT_PARSED_OUTPUT_PATH,
    DEFAULT_CHUNKED_OUTPUT_PATH,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_WORDS,
    EMBEDDING_MODEL_NAME,
    get_settings,
)
from src.ingestion.document_loader import load_and_parse_documents
from src.chunking.chunker import (
    load_jsonl,
    chunk_parsed_sections,
    save_jsonl,
)
from src.retrieval.vector_store import (
    ChromaVectorStore,
    build_vector_store_from_chunks,
)
from src.pipelines.rag_pipeline import RAGPipeline
from src.generation.paper_card_generator import (
    PaperCardGenerator,
    paper_card_to_markdown,
)
from src.generation.comparison_generator import (
    MultiPaperComparisonGenerator,
    comparison_to_markdown,
)
from src.generation.gap_finder import (
    ResearchGapFinder,
    gap_report_to_markdown,
)
from src.generation.reviewer import (
    ReviewerMode,
    review_to_markdown,
)

st.set_page_config(
    page_title="PaperMind",
    layout="wide",
)


def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --bg-soft: #111827;
            --panel: rgba(15, 23, 42, 0.92);
            --panel-2: rgba(17, 24, 39, 0.92);
            --border: #25314a;
            --text: #e5e7eb;
            --muted: #94a3b8;
            --accent: #4f46e5;
            --accent-2: #7c3aed;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(79,70,229,0.14), transparent 28%),
                radial-gradient(circle at top left, rgba(124,58,237,0.10), transparent 22%),
                linear-gradient(180deg, #0b1020 0%, #0b1020 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: #0a0f1c;
            border-right: 1px solid var(--border);
        }

        .brand-block {
            padding: 0.2rem 0 1rem 0;
        }

        .brand-title {
            font-size: 1.7rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .hero-card,
        .panel-card,
        .status-card,
        .info-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        }

        .hero-card {
            padding: 28px 30px;
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.5rem;
            color: var(--text);
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 0.4rem;
        }

        .pill {
            background: rgba(79, 70, 229, 0.14);
            border: 1px solid rgba(99, 102, 241, 0.28);
            color: #c7d2fe;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.85rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.25rem;
        }

        .section-subtitle {
            color: var(--muted);
            margin-bottom: 1.2rem;
            line-height: 1.6;
        }

        .mini-card {
            background: var(--panel-2);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px;
            min-height: 110px;
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.85rem;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .mini-value {
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 700;
        }

        .mini-small {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }

        .nav-caption {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.2rem;
            margin-bottom: 1rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1rem;
            font-weight: 600;
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.28);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            filter: brightness(1.05);
        }

        .stTextInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div,
        div[data-testid="stFileUploader"] section,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.92) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            color: var(--text) !important;
        }

        .stSlider [data-baseweb="slider"] {
            padding-top: 0.3rem;
        }

        .stRadio > div {
            gap: 0.45rem;
        }

        .stAlert {
            border-radius: 14px;
            border: 1px solid var(--border);
        }

        .stExpander {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }

        hr {
            border-color: var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">PaperMind</div>
            <div class="hero-subtitle">
                Research paper intelligence system for citation-grounded question answering,
                structured paper summaries, multi-paper comparison, research gap discovery,
                and reviewer-style analysis.
            </div>
            <div class="pill-row">
                <div class="pill">Citation-grounded RAG</div>
                <div class="pill">Paper Cards</div>
                <div class="pill">Comparison Matrix</div>
                <div class="pill">Research Gaps</div>
                <div class="pill">Reviewer Mode</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def get_indexed_files() -> list[str]:
    try:
        vector_store = ChromaVectorStore(
            persist_dir=CHROMA_DB_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )

        count = vector_store.count()
        if count == 0:
            return []

        results = vector_store.collection.get(
            include=["metadatas"],
            limit=count,
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

    except Exception:
        return []


def get_vector_store_count() -> int:
    try:
        vector_store = ChromaVectorStore(
            persist_dir=CHROMA_DB_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        return vector_store.count()
    except Exception:
        return 0


def save_uploaded_pdfs(uploaded_files) -> list[Path]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
            output_path = DOCS_DIR / uploaded_file.name
            with open(output_path, "wb") as file:
                file.write(uploaded_file.getbuffer())
            saved_paths.append(output_path)

    return saved_paths


def run_indexing_pipeline() -> dict:
    parsed_papers = load_and_parse_documents(
        input_dir=DOCS_DIR,
        output_path=DEFAULT_PARSED_OUTPUT_PATH,
    )

    parsed_records = load_jsonl(DEFAULT_PARSED_OUTPUT_PATH)

    chunks = chunk_parsed_sections(
        parsed_records=parsed_records,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        min_chunk_words=MIN_CHUNK_WORDS,
    )

    save_jsonl(
        records=chunks,
        output_path=DEFAULT_CHUNKED_OUTPUT_PATH,
    )

    vector_store = build_vector_store_from_chunks(
        chunks_path=DEFAULT_CHUNKED_OUTPUT_PATH,
        persist_dir=CHROMA_DB_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        reset=True,
    )

    return {
        "parsed_papers": len(parsed_papers),
        "parsed_sections": len(parsed_records),
        "chunks": len(chunks),
        "vector_store_count": vector_store.count(),
    }


def render_status_cards(indexed_files: list[str], vector_count: int):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">Indexed Papers</div>
                <div class="mini-value">{len(indexed_files)}</div>
                <div class="mini-small">Currently available in the vector store</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">Vector Chunks</div>
                <div class="mini-value">{vector_count}</div>
                <div class="mini-small">Embeddings stored in ChromaDB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-label">Document Folder</div>
                <div class="mini-value" style="font-size: 1rem;">{DOCS_DIR}</div>
                <div class="mini-small">Local source directory for uploaded PDFs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_error(error: Exception):
    st.error(str(error))
    with st.expander("Traceback"):
        st.code(traceback.format_exc())


def render_sources(sources: list[dict]):
    if not sources:
        st.info("No sources returned.")
        return

    with st.expander("Sources", expanded=False):
        for source in sources:
            st.markdown(
                f"""
**{source.get("source_id", "Source")}**  
Paper: `{source.get("paper_title", "Unknown")}`  
File: `{source.get("file_name", "Unknown")}`  
Page: `{source.get("page_number", "Unknown")}`  
Section: `{source.get("section_title", "Unknown")}`  
Citation: `{source.get("citation", "Unknown")}`  
Relevance: `{source.get("relevance_score", 0):.4f}`
"""
            )
            st.markdown("---")


def sidebar_panel(indexed_files: list[str], vector_count: int, settings):
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-block">
                <div class="brand-title">PaperMind</div>
                <div class="brand-subtitle">
                    Research workflow assistant for literature understanding and comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Workspace")
        page = st.radio(
            "Navigation",
            options=[
                "Upload and Index",
                "Ask Paper",
                "Paper Card",
                "Compare Papers",
                "Research Gaps",
                "Reviewer Mode",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div class='nav-caption'>Project modules and output generators</div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### System Status")
        st.markdown(
            f"""
            <div class="panel-card">
                <div style="color:#94a3b8;font-size:0.85rem;margin-bottom:0.35rem;">Indexed PDFs</div>
                <div style="font-size:1.65rem;font-weight:700;color:#e5e7eb;">{len(indexed_files)}</div>
                <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.6rem;">Vector chunks: {vector_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if indexed_files:
            st.markdown("**Available files**")
            for name in indexed_files:
                st.caption(name)

        st.markdown("---")
        st.markdown("### Runtime Configuration")

        llm_provider = st.selectbox(
            "LLM Provider",
            options=["ollama", "openai"],
            index=0 if settings.llm_provider == "ollama" else 1,
        )

        default_model = (
            settings.ollama_model_name
            if llm_provider == "ollama"
            else settings.openai_model_name
        )

        llm_model_name = st.text_input(
            "Model Name",
            value=default_model,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
        )

        top_k = st.slider(
            "Top K Retrieval",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )

        return page, llm_provider, llm_model_name, temperature, top_k


def page_upload_and_index():
    section_header(
        "Upload and Index Research Papers",
        "Store PDF files locally and rebuild the parsing, chunking, and vector indexing pipeline."
    )

    with st.form("save_uploads_form"):
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        save_submit = st.form_submit_button("Save Uploaded PDFs")

    if save_submit:
        try:
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                saved_paths = save_uploaded_pdfs(uploaded_files)
                st.success(f"Saved {len(saved_paths)} PDF file(s).")
                for path in saved_paths:
                    st.code(str(path))
        except Exception as error:
            show_error(error)

    st.markdown("---")

    with st.form("rebuild_index_form"):
        st.info(
            "This action parses all PDFs inside the docs directory, generates chunks, "
            "and rebuilds the Chroma vector store from scratch."
        )
        rebuild_submit = st.form_submit_button("Run Full Indexing Pipeline")

    if rebuild_submit:
        try:
            with st.spinner("Running parsing, chunking, embedding, and indexing pipeline..."):
                stats = run_indexing_pipeline()
            st.success("Indexing completed successfully.")
            st.json(stats)
        except Exception as error:
            show_error(error)


def page_ask_paper(indexed_files, llm_provider, llm_model_name, temperature, top_k):
    section_header(
        "Ask Paper",
        "Run citation-grounded question answering over one paper or over the entire indexed collection."
    )

    with st.form("ask_paper_form"):
        question = st.text_area(
            "Question",
            value="How does EvolveGCN use RNNs to evolve GCN parameters?",
            height=120,
        )

        file_filter = st.selectbox(
            "Optional file filter",
            options=["All indexed files"] + indexed_files,
        )

        submit = st.form_submit_button("Generate Answer")

    if submit:
        try:
            selected_file = None if file_filter == "All indexed files" else file_filter

            with st.spinner("Retrieving evidence and generating answer..."):
                pipeline = RAGPipeline(
                    top_k=top_k,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    temperature=temperature,
                )

                result = pipeline.ask(
                    question=question,
                    file_name=selected_file,
                )

            st.markdown("### Answer")
            st.markdown(result["answer"])

            st.markdown("### Runtime")
            st.write(f"Provider: `{result['llm_provider']}`")
            st.write(f"Model: `{result['model_name']}`")

            render_sources(result["sources"])

            with st.expander("Raw JSON"):
                st.json(result)

            st.download_button(
                "Download Answer JSON",
                data=json.dumps(result, indent=4, ensure_ascii=False),
                file_name="rag_answer.json",
                mime="application/json",
            )

        except Exception as error:
            show_error(error)


def page_paper_card(indexed_files, llm_provider, llm_model_name):
    section_header(
        "Paper Card",
        "Generate a structured summary card for a selected paper, including methods, datasets, results, and limitations."
    )

    if not indexed_files:
        st.info("No indexed papers available. Upload and index papers first.")
        return

    with st.form("paper_card_form"):
        selected_file = st.selectbox(
            "Select paper",
            options=indexed_files,
        )

        top_k = st.slider(
            "Top K per query",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            key="paper_card_top_k",
        )

        submit = st.form_submit_button("Generate Paper Card")

    if submit:
        try:
            with st.spinner("Generating Paper Card..."):
                generator = PaperCardGenerator(
                    top_k_per_query=top_k,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    temperature=0.1,
                )

                card = generator.generate(file_name=selected_file)
                markdown = paper_card_to_markdown(card)

            st.markdown(markdown)
            render_sources(card.sources)

            with st.expander("Raw JSON"):
                st.json(card.to_dict())

            st.download_button(
                "Download Paper Card Markdown",
                data=markdown,
                file_name=f"paper_card_{selected_file}.md",
                mime="text/markdown",
            )

            st.download_button(
                "Download Paper Card JSON",
                data=json.dumps(card.to_dict(), indent=4, ensure_ascii=False),
                file_name=f"paper_card_{selected_file}.json",
                mime="application/json",
            )

        except Exception as error:
            show_error(error)


def page_compare_papers(indexed_files, llm_provider, llm_model_name):
    section_header(
        "Compare Papers",
        "Create a multi-paper comparison matrix to contrast problem statements, methods, datasets, metrics, and findings."
    )

    if len(indexed_files) < 2:
        st.info("At least two indexed papers are required for comparison.")
        return

    with st.form("compare_form"):
        selected_files = st.multiselect(
            "Select papers",
            options=indexed_files,
            default=indexed_files[:2],
        )

        top_k = st.slider(
            "Top K per comparison query",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            key="compare_top_k",
        )

        submit = st.form_submit_button("Generate Comparison")

    if submit:
        try:
            if len(selected_files) < 2:
                st.warning("Please select at least two papers.")
                return

            with st.spinner("Generating comparison matrix..."):
                generator = MultiPaperComparisonGenerator(
                    top_k_per_query=top_k,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    temperature=0.1,
                )

                comparison = generator.generate(file_names=selected_files)
                markdown = comparison_to_markdown(comparison)

            st.markdown(markdown)
            render_sources(comparison.sources)

            with st.expander("Raw JSON"):
                st.json(comparison.to_dict())

            st.download_button(
                "Download Comparison Markdown",
                data=markdown,
                file_name="multi_paper_comparison.md",
                mime="text/markdown",
            )

            st.download_button(
                "Download Comparison JSON",
                data=json.dumps(comparison.to_dict(), indent=4, ensure_ascii=False),
                file_name="multi_paper_comparison.json",
                mime="application/json",
            )

        except Exception as error:
            show_error(error)


def page_research_gaps(indexed_files, llm_provider, llm_model_name):
    section_header(
        "Research Gaps",
        "Analyze selected papers and synthesize common limitations, evaluation gaps, dataset gaps, and future research directions."
    )

    if not indexed_files:
        st.info("No indexed papers available. Upload and index papers first.")
        return

    with st.form("gaps_form"):
        selected_files = st.multiselect(
            "Select papers",
            options=indexed_files,
            default=indexed_files[: min(2, len(indexed_files))],
        )

        top_k = st.slider(
            "Top K per gap query",
            min_value=1,
            max_value=3,
            value=1,
            step=1,
            key="gaps_top_k",
        )

        submit = st.form_submit_button("Generate Gap Report")

    if submit:
        try:
            if not selected_files:
                st.warning("Please select at least one paper.")
                return

            with st.spinner("Generating research gap report..."):
                finder = ResearchGapFinder(
                    top_k_per_query=top_k,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    temperature=0.2,
                )

                report = finder.generate(file_names=selected_files)
                markdown = gap_report_to_markdown(report)

            st.markdown(markdown)
            render_sources(report.sources)

            with st.expander("Raw JSON"):
                st.json(report.to_dict())

            st.download_button(
                "Download Gap Report Markdown",
                data=markdown,
                file_name="research_gap_report.md",
                mime="text/markdown",
            )

            st.download_button(
                "Download Gap Report JSON",
                data=json.dumps(report.to_dict(), indent=4, ensure_ascii=False),
                file_name="research_gap_report.json",
                mime="application/json",
            )

        except Exception as error:
            show_error(error)


def page_reviewer_mode(indexed_files, llm_provider, llm_model_name):
    section_header(
        "Reviewer Mode",
        "Generate a reviewer-style critique with strengths, weaknesses, novelty assessment, missing experiments, and recommendation."
    )

    if not indexed_files:
        st.info("No indexed papers available. Upload and index papers first.")
        return

    with st.form("review_form"):
        selected_file = st.selectbox(
            "Select paper",
            options=indexed_files,
        )

        top_k = st.slider(
            "Top K per review query",
            min_value=1,
            max_value=3,
            value=1,
            step=1,
            key="review_top_k",
        )

        submit = st.form_submit_button("Generate Reviewer Report")

    if submit:
        try:
            with st.spinner("Generating reviewer report..."):
                reviewer = ReviewerMode(
                    top_k_per_query=top_k,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    temperature=0.2,
                )

                review = reviewer.generate(file_name=selected_file)
                markdown = review_to_markdown(review)

            st.markdown(markdown)
            render_sources(review.sources)

            with st.expander("Raw JSON"):
                st.json(review.to_dict())

            st.download_button(
                "Download Reviewer Report Markdown",
                data=markdown,
                file_name=f"reviewer_report_{selected_file}.md",
                mime="text/markdown",
            )

            st.download_button(
                "Download Reviewer Report JSON",
                data=json.dumps(review.to_dict(), indent=4, ensure_ascii=False),
                file_name=f"reviewer_report_{selected_file}.json",
                mime="application/json",
            )

        except Exception as error:
            show_error(error)


def main():
    inject_css()

    settings = get_settings()
    indexed_files = get_indexed_files()
    vector_count = get_vector_store_count()

    page, llm_provider, llm_model_name, temperature, top_k = sidebar_panel(
        indexed_files=indexed_files,
        vector_count=vector_count,
        settings=settings,
    )

    render_hero()
    render_status_cards(indexed_files=indexed_files, vector_count=vector_count)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    if page == "Upload and Index":
        page_upload_and_index()

    elif page == "Ask Paper":
        page_ask_paper(
            indexed_files=indexed_files,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            temperature=temperature,
            top_k=top_k,
        )

    elif page == "Paper Card":
        page_paper_card(
            indexed_files=indexed_files,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
        )

    elif page == "Compare Papers":
        page_compare_papers(
            indexed_files=indexed_files,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
        )

    elif page == "Research Gaps":
        page_research_gaps(
            indexed_files=indexed_files,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
        )

    elif page == "Reviewer Mode":
        page_reviewer_mode(
            indexed_files=indexed_files,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
        )


if __name__ == "__main__":
    main()