"""
FAISS index wrapper (Repository pattern) - persists to disk so the index
survives restarts. Stores a parallel id-map so FAISS's integer positions
resolve back to our chunk UUIDs.
"""
import os
import json
import threading
import faiss
import numpy as np
from app.config import get_settings

settings = get_settings()


class FaissStore:
    def __init__(self, dim: int = 384, index_dir: str | None = None):
        self.dim = dim
        self.index_dir = index_dir or settings.faiss_index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.ids_path = os.path.join(self.index_dir, "ids.json")
        # Guards index.add() + id_map.extend() + disk persist as one atomic
        # unit. Without this, two concurrent background jobs calling add()
        # could interleave in a way that desyncs id_map ordering from the
        # FAISS index's internal vector ordering - silent, hard-to-debug
        # corruption rather than a crash. A 10-thread stress test didn't
        # reproduce it, but that's GIL timing luck, not a guarantee, given
        # this is a singleton written from concurrent async background tasks.
        self._lock = threading.Lock()

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.ids_path) as f:
                self.id_map: list[str] = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine sim via normalized vectors
            self.id_map = []

    def add(self, vectors: np.ndarray, chunk_ids: list[str]):
        with self._lock:
            self.index.add(vectors.astype("float32"))
            self.id_map.extend(chunk_ids)
            self._persist()

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            scores, indices = self.index.search(query_vector.reshape(1, -1).astype("float32"), top_k)
            id_map_snapshot = self.id_map  # safe: list reference, not mutated in place elsewhere
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((id_map_snapshot[idx], float(score)))
        return results

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "w") as f:
            json.dump(self.id_map, f)


_store: FaissStore | None = None


def get_faiss_store() -> FaissStore:
    global _store
    if _store is None:
        _store = FaissStore()
    return _store
