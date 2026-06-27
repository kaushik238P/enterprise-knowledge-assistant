from dataclasses import dataclass
from typing import Optional
import logging

from fastembed import SparseTextEmbedding

from backend.config.settings import settings
from backend.ingestion.metadata import ChunkData

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEmbedder:
    def __init__(
        self,
        model_name: str = settings.sparse_model,
        batch_size: int = settings.embedding_batch_size,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer. Got: {batch_size}")

        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Optional[SparseTextEmbedding] = None

    def _load_model(self) -> SparseTextEmbedding:
        if self._model is not None:
            return self._model

        try:
            self._model = SparseTextEmbedding(model_name=self._model_name)
            logger.info("Sparse model loaded | model=%s", self._model_name)
        except Exception as exc:
            logger.exception("Failed to load sparse model | model=%s | error=%s", self._model_name, exc)
            raise RuntimeError(f"Failed to load sparse embedding model '{self._model_name}': {exc}") from exc

        return self._model

    def embed_text(self, text: str) -> SparseVector:
        if not text or not text.strip():
            raise ValueError("text cannot be empty or whitespace-only.")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            raise ValueError("texts list cannot be empty.")

        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty_indices:
            raise ValueError(f"texts at indices {empty_indices} are empty or whitespace-only.")

        model = self._load_model()

        try:
            raw_embeddings = list(model.embed(texts))
        except Exception as exc:
            logger.exception(
                "Sparse encoding failed | count=%d | model=%s | error=%s",
                len(texts),
                self._model_name,
                exc,
            )
            raise RuntimeError(
                f"Sparse embedding failed for {len(texts)} texts: {exc}"
            ) from exc

        if len(raw_embeddings) != len(texts):
            raise RuntimeError(
                f"Sparse embedding count mismatch. Expected {len(texts)}, got {len(raw_embeddings)}."
            )

        embeddings: list[SparseVector] = []
        for embedding in raw_embeddings:
            if len(embedding.indices) != len(embedding.values):
                raise RuntimeError("Sparse embedding indices/values length mismatch.")

            embeddings.append(
                SparseVector(
                    indices=list(embedding.indices),
                    values=list(embedding.values),
                )
            )

        logger.info("Sparse embeddings generated | count=%d", len(embeddings))
        return embeddings

    def embed_chunks(self, chunks: list[ChunkData]) -> list[SparseVector]:
        if not chunks:
            raise ValueError("chunks list cannot be empty.")

        empty_indices = [
            i for i, c in enumerate(chunks)
            if not c.content or not c.content.strip()
        ]
        if empty_indices:
            raise ValueError(f"ChunkData at indices {empty_indices} have empty content.")

        return self.embed_texts([chunk.content for chunk in chunks])

    def embed_query(self, query: str) -> SparseVector:
        return self.embed_text(query)

    def is_loaded(self) -> bool:
        return self._model is not None


_DEFAULT_SPARSE_EMBEDDER: SparseEmbedder | None = None


def get_sparse_embedder() -> SparseEmbedder:
    global _DEFAULT_SPARSE_EMBEDDER

    if _DEFAULT_SPARSE_EMBEDDER is None:
        _DEFAULT_SPARSE_EMBEDDER = SparseEmbedder()

    return _DEFAULT_SPARSE_EMBEDDER