import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vectorstore.bm25_store import BM25Store


def test_bm25_ranks_exact_keyword_match_highest():
    store = BM25Store()
    chunk_ids = ["a", "b", "c"]
    texts = [
        "The UPI transaction limit is increased to Rs 1000",
        "KYC verification requires Aadhaar based e-KYC",
        "Loan interest rate disclosure is mandatory",
    ]
    store.build(chunk_ids, texts)
    results = store.search("UPI transaction limit", top_k=3)
    assert results[0][0] == "a"  # exact keyword match should rank first


def test_bm25_empty_store_returns_empty():
    store = BM25Store()
    assert store.search("anything", top_k=5) == []
