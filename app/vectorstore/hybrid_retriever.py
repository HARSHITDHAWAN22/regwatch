"""
Fuses dense (FAISS) + sparse (BM25) retrieval via Reciprocal Rank Fusion.
This is standard production RAG practice - dense catches semantic
similarity, sparse catches exact terms/numbers/section refs that
embeddings often smooth over. Neither alone is reliable for legal text.
"""
from app.embeddings.embedder import embed_text
from app.vectorstore.faiss_store import get_faiss_store
from app.vectorstore.bm25_store import get_bm25_store

RRF_K = 60  # standard constant from the RRF paper - dampens the impact of any single rank


def hybrid_search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Returns [(chunk_id, fused_score), ...] sorted best-first."""
    dense_results = get_faiss_store().search(embed_text(query), top_k=top_k)
    sparse_results = get_bm25_store().search(query, top_k=top_k)

    fused_scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(dense_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)

    for rank, (chunk_id, _) in enumerate(sparse_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
