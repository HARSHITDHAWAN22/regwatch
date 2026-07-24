"""
Tests concurrent processing of multiple circulars - the FAISS store and
BM25 store are module-level singletons shared across background tasks.
FastAPI's BackgroundTasks run sequentially within a single request/response
cycle, but real usage means multiple *concurrent* uploads (different HTTP
requests) could still trigger overlapping background jobs. This checks
whether that's actually safe.
"""
import sys, os, shutil, threading, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["DATABASE_URL"] = "sqlite:///./concurrency_test.db"
os.environ["FAISS_INDEX_DIR"] = "./data/concurrency_test_faiss"
if os.path.exists("concurrency_test.db"):
    os.remove("concurrency_test.db")
if os.path.exists("./data/concurrency_test_faiss"):
    shutil.rmtree("./data/concurrency_test_faiss")

import numpy as np, types
fake_st = types.ModuleType("sentence_transformers")
class FakeST:
    def __init__(self, *a, **kw): pass
    def encode(self, texts, **kw):
        rng = np.random.default_rng(3)
        return np.array([rng.random(384).astype("float32") for _ in texts])
class FakeCE:
    def __init__(self, *a, **kw): pass
    def predict(self, pairs):
        return np.array([0.9] * len(pairs))
fake_st.SentenceTransformer = FakeST
fake_st.CrossEncoder = FakeCE
sys.modules["sentence_transformers"] = fake_st

import app.reasoning.llm_client as llm_client_module
def fake_call_llm_json(system_prompt, user_prompt):
    if "applies_to" in system_prompt:
        return ({"applies_to": [], "effective_date": None, "action_required": False,
                  "penalty_mentioned": False, "references_other_circulars": []}, 50, 50)
    if "strict auditor" in system_prompt:
        return ({"score": 4, "critique": "ok"}, 30, 30)
    return ({"impacts_policy": True, "severity": "info", "reasoning": "test reasoning",
              "evidence_sentence": "test"}, 50, 50)
llm_client_module.call_llm_json = fake_call_llm_json
import app.reasoning.impact_reasoner as ir, app.reasoning.verifier as vf, app.ingestion.structured_extractor as se
ir.call_llm_json = fake_call_llm_json
vf.call_llm_json = fake_call_llm_json
se.call_llm_json = fake_call_llm_json

from app.db import SessionLocal, init_db
from app.models.policy import Policy
from app.pipeline import ingest_circular, run_impact_assessment

init_db()

db = SessionLocal()
db.add(Policy(name="Test Policy", description="A policy for concurrency testing."))
db.commit()
db.close()

os.makedirs("/tmp/concurrencytest", exist_ok=True)
for i in range(10):
    with open(f"/tmp/concurrencytest/circular_{i}.txt", "w") as f:
        f.write(f"1. This is clause one of circular {i} about test policy matters.\n\n"
                f"2. This is clause two of circular {i} with different content entirely.")

results = {"passed": 0, "failed": 0}
errors = []

def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    results["passed" if cond else "failed"] += 1
    print(f"[{status}] {name} {extra}")


def process_circular(i):
    try:
        db = SessionLocal()
        circular = ingest_circular(db, f"/tmp/concurrencytest/circular_{i}.txt", f"Circular {i}", f"C-{i}")
        run_impact_assessment(db, circular)
        db.close()
    except Exception as e:
        errors.append((i, str(e)))


print("\n--- Concurrent circular processing (10 threads) ---")
threads = [threading.Thread(target=process_circular, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

check("No exceptions raised during concurrent processing", len(errors) == 0, f"errors={errors}")

# Verify data integrity after concurrent writes
db = SessionLocal()
from app.models.circular import Circular, CircularChunk
circulars = db.query(Circular).all()
check("All 10 circulars were persisted", len(circulars) == 10, f"count={len(circulars)}")

total_chunks = db.query(CircularChunk).count()
check("Chunks from all circulars persisted (20 = 10 circulars x 2 clauses)", total_chunks == 20, f"count={total_chunks}")

# Verify FAISS index has all vectors and no corruption
from app.vectorstore.faiss_store import get_faiss_store
faiss_store = get_faiss_store()
check("FAISS index vector count matches chunk count", faiss_store.index.ntotal == total_chunks,
      f"faiss_ntotal={faiss_store.index.ntotal}, db_chunks={total_chunks}")
check("FAISS id_map length matches vector count", len(faiss_store.id_map) == faiss_store.index.ntotal,
      f"id_map_len={len(faiss_store.id_map)}, ntotal={faiss_store.index.ntotal}")

# Verify no duplicate/corrupted chunk_ids in the id_map (would indicate a race condition)
check("No duplicate chunk_ids in FAISS id_map (would indicate a race)",
      len(faiss_store.id_map) == len(set(faiss_store.id_map)), f"unique={len(set(faiss_store.id_map))}, total={len(faiss_store.id_map)}")

db.close()

print(f"\n{'='*50}")
print(f"CONCURRENCY TEST: {results['passed']} passed, {results['failed']} failed")
print(f"{'='*50}")

os.remove("concurrency_test.db")
shutil.rmtree("/tmp/concurrencytest")
shutil.rmtree("./data/concurrency_test_faiss", ignore_errors=True)
sys.exit(1 if results["failed"] > 0 else 0)
