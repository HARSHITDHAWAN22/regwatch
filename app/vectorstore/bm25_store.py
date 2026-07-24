"""
Sparse (keyword) retrieval via BM25 - catches exact term/number matches
(e.g. 'Rs. 10,000', 'Section 4.2') that dense embeddings often blur over.
Rebuilt in-memory from all chunks; fine at this project's scale (thousands
of chunks). For a larger corpus you'd swap this for Elasticsearch/OpenSearch
behind the same interface - that's the point of keeping it behind a class.
"""
from rank_bm25 import BM25Okapi
import threading


class BM25Store:
    def __init__(self):
        self.chunk_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._lock = threading.Lock()  # see FaissStore for why this matters

    def build(self, chunk_ids: list[str], texts: list[str]):
        with self._lock:
            self.chunk_ids = chunk_ids
            self.tokenized_corpus = [t.lower().split() for t in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def add(self, chunk_ids: list[str], texts: list[str]):
        """Incrementally add and rebuild (BM25Okapi has no true incremental add)."""
        with self._lock:
            self.chunk_ids.extend(chunk_ids)
            self.tokenized_corpus.extend([t.lower().split() for t in texts])
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            if self.bm25 is None:
                return []
            bm25, chunk_ids = self.bm25, self.chunk_ids  # snapshot references under lock
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


_store: BM25Store | None = None


def get_bm25_store() -> BM25Store:
    global _store
    if _store is None:
        _store = BM25Store()
    return _store
