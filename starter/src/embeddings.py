
"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Local embedding adapter using Sentence Transformers."""

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")

        self.model_name = model_name

        # Load the actual pretrained embedding model
        self.model = SentenceTransformer(model_name)

        # Get the model's actual embedding dimensions
        self.dimensions = self.model.get_embedding_dimension()

        if self.dimensions is None:
            raise ValueError("Cannot determine embedding dimensions")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate one normalized embedding vector per document."""

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All document texts must be non-empty strings")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return [
            [float(value) for value in vector]
            for vector in embeddings
        ]

    def embed_query(self, question: str) -> list[float]:
        """Generate a normalized query embedding."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        embedding = self.model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return [float(value) for value in embedding]
