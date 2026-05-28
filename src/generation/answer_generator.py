from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json
import urllib.request
import urllib.error

from openai import OpenAI

from src.config.settings import get_settings
from src.retrieval.retriever import RetrievedChunk


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict]
    retrieved_context_count: int
    llm_provider: str
    model_name: str

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerGenerator:
    """
    Generates citation-grounded answers from retrieved research paper chunks.

    Supported providers:
        - ollama: local free LLM
        - openai: cloud API

    Retriever finds evidence.
    Generator writes the answer from that evidence.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        provider: Optional[str] = None,
    ):
        settings = get_settings()

        self.provider = (provider or settings.llm_provider).lower()
        self.temperature = temperature

        if self.provider == "ollama":
            self.model_name = model_name or settings.ollama_model_name
            self.ollama_base_url = settings.ollama_base_url.rstrip("/")
            self.ollama_timeout_seconds = settings.ollama_timeout_seconds
            self.openai_client = None

        elif self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY was not found. "
                    "Either add OPENAI_API_KEY to .env or set LLM_PROVIDER=ollama."
                )

            self.model_name = model_name or settings.openai_model_name
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.ollama_base_url = None
            self.ollama_timeout_seconds = None

        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                "Use 'ollama' or 'openai'."
            )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RAGAnswer:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not retrieved_chunks:
            return RAGAnswer(
                question=question,
                answer=(
                    "I could not find relevant context in the indexed papers "
                    "to answer this question."
                ),
                sources=[],
                retrieved_context_count=0,
                llm_provider=self.provider,
                model_name=self.model_name,
            )

        context = self._format_context(retrieved_chunks)
        prompt = self._build_prompt(question=question, context=context)

        if self.provider == "ollama":
            answer_text = self._generate_with_ollama(prompt)
        elif self.provider == "openai":
            answer_text = self._generate_with_openai(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        sources = self._build_sources(retrieved_chunks)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieved_context_count=len(retrieved_chunks),
            llm_provider=self.provider,
            model_name=self.model_name,
        )

    def _generate_with_openai(self, prompt: str) -> str:
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.temperature,
        )

        return response.choices[0].message.content

    def _generate_with_ollama(self, prompt: str) -> str:
        """
        Call Ollama local chat API.

        Ollama must be running locally:
            ollama serve

        And the model must exist:
            ollama pull qwen3:14b
        """
        url = f"{self.ollama_base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.ollama_timeout_seconds,
            ) as response:
                response_data = response.read().decode("utf-8")
                result = json.loads(response_data)

        except urllib.error.URLError as error:
            raise RuntimeError(
                "Failed to connect to Ollama. Make sure Ollama is running. "
                "Try running: ollama serve"
            ) from error

        if "message" not in result or "content" not in result["message"]:
            raise RuntimeError(f"Unexpected Ollama response: {result}")

        return result["message"]["content"]

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are PaperMind, a research paper analysis assistant. "
            "Answer only using the provided context. "
            "If the context is insufficient, clearly say that the retrieved context "
            "does not contain enough evidence. "
            "Use source labels like [Source 1], [Source 2] when making claims. "
            "Do not fabricate paper details, datasets, results, equations, or conclusions."
        )

    def _format_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        context_blocks = []

        for chunk in retrieved_chunks:
            block = (
                f"[Source {chunk.rank}]\n"
                f"Paper: {chunk.paper_title}\n"
                f"File: {chunk.file_name}\n"
                f"Page: {chunk.page_number}\n"
                f"Section: {chunk.section_title}\n"
                f"Citation: {chunk.source}\n"
                f"Relevance Score: {chunk.relevance_score:.4f}\n"
                f"Text:\n{chunk.text}"
            )

            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)

    def _build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        return f"""
You are given retrieved context from research papers.

Your task:
1. Answer the user's question using only the provided context.
2. Cite evidence using [Source 1], [Source 2], etc.
3. Do not invent paper details that are not in the context.
4. If the retrieved context is incomplete, say what is missing.
5. Prefer a clear research-style explanation.

Question:
{question}

Retrieved Context:
{context}

Answer:
""".strip()

    def _build_sources(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[dict]:
        sources = []

        for chunk in retrieved_chunks:
            sources.append(
                {
                    "source_id": f"Source {chunk.rank}",
                    "paper_title": chunk.paper_title,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "citation": chunk.source,
                    "relevance_score": chunk.relevance_score,
                    "chunk_id": chunk.chunk_id,
                }
            )

        return sources

    @staticmethod
    def print_answer(rag_answer: RAGAnswer):
        print("\nQUESTION")
        print("=" * 100)
        print(rag_answer.question)

        print("\nANSWER")
        print("=" * 100)
        print(rag_answer.answer)

        print("\nSOURCES")
        print("=" * 100)
        print(json.dumps(rag_answer.sources, indent=4, ensure_ascii=False))