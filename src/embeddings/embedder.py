from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """
    Wrapper around SentenceTransformer for creating text embeddings.

    This wrapper keeps embedding logic separate from vector database logic.
    That makes it easier to replace the embedding model later.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        """
        Convert a list of texts into embeddings.

        Args:
            texts:
                List of chunk texts.

            batch_size:
                Batch size for embedding generation.

            normalize_embeddings:
                If True, embeddings are L2-normalized. This is useful for cosine-style similarity.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
        )

        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(float).tolist()

        return embeddings

    def embed_query(
        self,
        query: str,
        normalize_embeddings: bool = True,
    ) -> list[float]:
        """
        Embed one user query.
        """
        embedding = self.model.encode(
            query,
            normalize_embeddings=normalize_embeddings,
        )

        if isinstance(embedding, np.ndarray):
            return embedding.astype(float).tolist()

        return embedding