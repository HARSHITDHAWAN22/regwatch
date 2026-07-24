"""
Thin wrapper around Sentence-Transformers so the rest of the app never
imports the library directly (Repository-style abstraction - swap the
embedding model/provider here without touching callers).
"""
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import get_settings


@lru_cache
def _get_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
