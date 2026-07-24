"""
Cross-encoder reranker - re-scores the hybrid-retrieval candidates by
jointly encoding (query, candidate) pairs, which is far more accurate
than comparing independently-computed embeddings. This is the step most
student RAG projects skip entirely; it's what production systems use to
turn "roughly relevant top-20" into "precisely relevant top-5".
"""
from functools import lru_cache
from sentence_transformers import CrossEncoder
from app.config import get_settings


@lru_cache
def _get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 5) -> list[tuple[str, str, float]]:
    """
    candidates: [(chunk_id, chunk_text), ...]
    returns: [(chunk_id, chunk_text, rerank_score), ...] sorted best-first, truncated to top_k
    """
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, text) for _, text in candidates]
    scores = model.predict(pairs)
    scored = [(cid, text, float(score)) for (cid, text), score in zip(candidates, scores)]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]
