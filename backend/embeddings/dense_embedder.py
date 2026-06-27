from typing import Optional
import logging

from sentence_transformers import SentenceTransformer

from backend.config.settings import settings
from backend.ingestion.metadata import ChunkData

logger = logging.getLogger(__name__)


class DenseEmbedder:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        batch_size: int = settings.embedding_batch_size,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer. Got: {batch_size}")

        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Optional[SentenceTransformer] = None
        self._embedding_dimension: Optional[int] = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        try:
            self._model = SentenceTransformer(self._model_name)
            self._embedding_dimension = self._model.get_embedding_dimension()
            if self._embedding_dimension != settings.embedding_dimension:
                raise RuntimeError(
                    f"Embedding dimension mismatch. "
                    f"Model returned {self._embedding_dimension}, "
                    f"settings expect {settings.embedding_dimension}."
                )

            logger.info(
                "Dense model loaded | model=%s | dimension=%d",
                self._model_name,
                self._embedding_dimension,
            )
        except Exception as exc:
            logger.exception("Failed to load dense model | model=%s | error=%s", self._model_name, exc)
            raise RuntimeError(f"Failed to load SentenceTransformer model '{self._model_name}': {exc}") from exc

        return self._model

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text cannot be empty or whitespace-only.")
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts list cannot be empty.")

        empty_indices = [i for i, t in enumerate(texts) if not t or not t.strip()]
        if empty_indices:
            raise ValueError(f"texts at indices {empty_indices} are empty or whitespace-only.")

        model = self._load_model()

        raw_embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if raw_embeddings is None or len(raw_embeddings) != len(texts):
            raise RuntimeError("Dense embedding output shape mismatch.")

        logger.info("Dense embedding complete | count=%d", len(texts))

        return [embedding.tolist() for embedding in raw_embeddings]

    def embed_chunks(self, chunks: list[ChunkData]) -> list[list[float]]:
        if not chunks:
            raise ValueError("chunks list cannot be empty.")

        empty_indices = [
            i for i, c in enumerate(chunks)
            if not c.content or not c.content.strip()
        ]
        if empty_indices:
            raise ValueError(f"ChunkData at indices {empty_indices} have empty content.")

        return self.embed_texts([chunk.content for chunk in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)

    def get_embedding_dimension(self) -> int:
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        self._load_model()
        assert self._embedding_dimension is not None
        return self._embedding_dimension


_DEFAULT_DENSE_EMBEDDER = DenseEmbedder()


def get_dense_embedder() -> DenseEmbedder:
    return _DEFAULT_DENSE_EMBEDDER