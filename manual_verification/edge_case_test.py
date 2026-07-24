"""
Edge-case / stress test covering paths the prior integration tests didn't
touch: FAISS persistence across a process restart, LRU cache eviction at
actual capacity, RBAC denial, circuit breaker tripping + recovery,
pathological chunker input, and 404/error handling.
"""
import sys, os, shutil, types
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# MUST be set before any `app.*` import, since app.config.get_settings() is
# @lru_cache'd - the first import anywhere in the process that touches
# settings freezes DATABASE_URL/FAISS_INDEX_DIR for the rest of the run.
# Setting these later (as an earlier version of this script did) silently
# falls back to the real default regwatch.db instead of an isolated test DB.
os.environ["DATABASE_URL"] = "sqlite:///./edge_test.db"
os.environ["FAISS_INDEX_DIR"] = "./data/edge_test_faiss2"
if os.path.exists("edge_test.db"):
    os.remove("edge_test.db")
if os.path.exists("./data/edge_test_faiss2"):
    shutil.rmtree("./data/edge_test_faiss2")

import numpy as np

fake_st = types.ModuleType("sentence_transformers")
class FakeST:
    def __init__(self, *a, **kw): pass
    def encode(self, texts, **kw):
        rng = np.random.default_rng(7)
        return np.array([rng.random(384).astype("float32") for _ in texts])
class FakeCE:
    def __init__(self, *a, **kw): pass
    def predict(self, pairs):
        return np.array([0.9] * len(pairs))
fake_st.SentenceTransformer = FakeST
fake_st.CrossEncoder = FakeCE
sys.modules["sentence_transformers"] = fake_st

results = {"passed": 0, "failed": 0}
def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    results["passed" if cond else "failed"] += 1
    print(f"[{status}] {name} {extra}")


# ============================================================
# 1. FAISS persistence across a simulated process restart
# ============================================================
print("\n--- FAISS persistence across restart ---")
test_faiss_dir = "./data/edge_test_faiss"
if os.path.exists(test_faiss_dir):
    shutil.rmtree(test_faiss_dir)

from app.vectorstore.faiss_store import FaissStore

store1 = FaissStore(dim=384, index_dir=test_faiss_dir)
vectors = np.random.default_rng(1).random((3, 384)).astype("float32")
store1.add(vectors, ["chunk-a", "chunk-b", "chunk-c"])
check("Index has 3 vectors after add", store1.index.ntotal == 3, f"ntotal={store1.index.ntotal}")

# Simulate a fresh process: new FaissStore instance pointed at the same dir
store2 = FaissStore(dim=384, index_dir=test_faiss_dir)
check("Reloaded index has same vector count", store2.index.ntotal == 3, f"ntotal={store2.index.ntotal}")
check("Reloaded id_map matches original", store2.id_map == ["chunk-a", "chunk-b", "chunk-c"], f"id_map={store2.id_map}")

results_search = store2.search(vectors[0], top_k=1)
check("Search on reloaded index finds the exact vector back", results_search[0][0] == "chunk-a",
      f"results={results_search}")
shutil.rmtree(test_faiss_dir)


# ============================================================
# 2. LRU cache eviction at actual capacity
# ============================================================
print("\n--- LRU cache eviction at capacity ---")
from app.cache import cached_assessment, clear_cache, cache_stats, _CACHE_MAX_SIZE

clear_cache()

@cached_assessment
def dummy(clause_text, policy_id, prompt_version):
    return {"value": clause_text}

# Fill beyond capacity
n_to_add = _CACHE_MAX_SIZE + 50
for i in range(n_to_add):
    dummy(f"clause-{i}", "policy-1", "v1")

stats = cache_stats()
check("Cache size capped at max, not growing unbounded", stats["size"] == _CACHE_MAX_SIZE,
      f"size={stats['size']}, max={_CACHE_MAX_SIZE}")

# The earliest entries should have been evicted (LRU) - re-adding clause-0 should be a fresh compute
r = dummy("clause-0", "policy-1", "v1")
check("Oldest entry was evicted (re-computed, not cache hit)", r["was_cache_hit"] is False, f"result={r}")

# The most recently added entries should still be cached
r = dummy(f"clause-{n_to_add - 1}", "policy-1", "v1")
check("Most recent entry still cached (cache hit)", r["was_cache_hit"] is True, f"result={r}")
clear_cache()


# ============================================================
# 3. Circuit breaker trips after repeated failures, recovers after cooldown
# ============================================================
print("\n--- Circuit breaker trip + recovery ---")
from app.reasoning.llm_client import CircuitBreaker

breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=1)
check("Breaker starts closed", breaker.is_open() is False)

for _ in range(3):
    breaker.record_failure()
check("Breaker opens after failure_threshold consecutive failures", breaker.is_open() is True)

import time
time.sleep(1.1)
check("Breaker closes again after cooldown elapses", breaker.is_open() is False)

breaker.record_failure()
breaker.record_failure()
breaker.record_success()
check("A success resets the failure count", breaker.failure_count == 0, f"failure_count={breaker.failure_count}")


# ============================================================
# 4. Chunker pathological input
# ============================================================
print("\n--- Chunker pathological input ---")
from app.ingestion.chunker import get_chunker

chunker = get_chunker("section")
check("Empty string produces no chunks (no crash)", chunker.chunk("") == [])
check("Whitespace-only produces no chunks (no crash)", chunker.chunk("   \n\n   ") == [])

huge_text = "1. " + ("word " * 50000)  # ~50k words, single numbered section
chunks = chunker.chunk(huge_text)
check("Very large single-section text doesn't crash and produces 1 chunk", len(chunks) == 1, f"n_chunks={len(chunks)}")

unicode_text = "1. यह एक परीक्षण खंड है जिसमें RBI परिपत्र शामिल है।\n\n2. Second clause with émojis 🏦💰 and spëcial chars."
chunks = chunker.chunk(unicode_text)
check("Unicode/non-ASCII text doesn't crash", len(chunks) == 2, f"n_chunks={len(chunks)}")


# ============================================================
# 5. RBAC denial + 404 handling via real FastAPI app
# ============================================================
print("\n--- RBAC denial + error handling (real HTTP) ---")
import app.reasoning.llm_client as llm_client_module
def fake_call_llm_json(system_prompt, user_prompt):
    return ({"impacts_policy": False, "severity": "info", "reasoning": "n/a", "evidence_sentence": ""}, 10, 10)
llm_client_module.call_llm_json = fake_call_llm_json
import app.reasoning.impact_reasoner as ir, app.reasoning.verifier as vf, app.ingestion.structured_extractor as se
ir.call_llm_json = fake_call_llm_json
vf.call_llm_json = fake_call_llm_json
se.call_llm_json = fake_call_llm_json

from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal, init_db
from app.models.user import User, Role
from app.auth import hash_password

init_db()
client = TestClient(app)

db = SessionLocal()
db.add(User(email="viewer@test.com", hashed_password=hash_password("pw"), role=Role.VIEWER))
db.commit()
db.close()

r = client.post("/auth/login", json={"email": "viewer@test.com", "password": "pw"})
viewer_token = r.json()["access_token"]
viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

r = client.post("/policies", headers=viewer_headers, json={"name": "X", "description": "Y"})
check("Viewer role CANNOT create a policy (RBAC denies)", r.status_code == 403, f"status={r.status_code}, body={r.json()}")

r = client.post("/impacts/nonexistent-id/review", headers=viewer_headers, json={"review_status": "confirmed"})
check("Reviewing a nonexistent impact returns 403 or 404, not 500",
      r.status_code in (403, 404), f"status={r.status_code}")

r = client.get("/jobs/nonexistent-job-id")
check("Polling a nonexistent job_id returns 404, not 500", r.status_code == 404, f"status={r.status_code}")

r = client.post("/auth/login", json={"email": "viewer@test.com", "password": "WRONG_PASSWORD"})
check("Wrong password correctly rejected", r.status_code == 401)

r = client.get("/impacts", headers={"Authorization": "Bearer garbage.invalid.token"})
check("Garbage JWT on a public endpoint doesn't crash the server",
      r.status_code in (200, 401), f"status={r.status_code}")

# ============================================================
# 6. Auth password-length validation (bcrypt 72-byte silent-truncation bug)
# ============================================================
print("\n--- Password length validation (bcrypt 72-byte limit) ---")
r = client.post("/auth/register", json={"email": "longpw@test.com", "password": "x" * 100, "role": "viewer"})
check("Registration REJECTS a password over 72 bytes (was previously silently truncated)",
      r.status_code == 422, f"status={r.status_code}, body={r.json()}")

r = client.post("/auth/register", json={"email": "shortpw@test.com", "password": "1234567", "role": "viewer"})
check("Registration REJECTS a password under 8 characters", r.status_code == 422, f"status={r.status_code}")

r = client.post("/auth/register", json={"email": "okpw@test.com", "password": "ValidPassword123", "role": "viewer"})
check("Registration ACCEPTS a reasonable password", r.status_code == 200, f"status={r.status_code}, body={r.json()}")

r2 = client.post("/auth/register", json={"email": "okpw@test.com", "password": "AnotherValidPw1", "role": "viewer"})
check("Registering a duplicate email is rejected", r2.status_code == 400, f"status={r2.status_code}")

r3 = client.post("/auth/register", json={"email": "badrole@test.com", "password": "ValidPassword123", "role": "superadmin"})
check("Registration REJECTS an invalid role string", r3.status_code == 400, f"status={r3.status_code}, body={r3.json()}")


# ============================================================
# 7. Empty-state pipeline behavior (zero policies registered)
# ============================================================
print("\n--- Empty-state pipeline (zero policies in registry) ---")
from app.pipeline import ingest_circular, run_impact_assessment
from app.db import SessionLocal

db2 = SessionLocal()
with open("/tmp/empty_state_circular.txt", "w") as f:
    f.write("1. Some clause text that would normally be checked against policies.")
circular_no_policies = ingest_circular(db2, "/tmp/empty_state_circular.txt", "No-Policy-Match Circular")
try:
    assessments = run_impact_assessment(db2, circular_no_policies)
    check("run_impact_assessment doesn't crash when there ARE policies but this is a fresh DB check", True)
except Exception as e:
    check("run_impact_assessment doesn't crash", False, f"exception={e}")
db2.close()
os.remove("/tmp/empty_state_circular.txt")


# ============================================================
# 8. Whitespace-only circular upload (zero chunks produced)
# ============================================================
print("\n--- Whitespace-only circular (zero chunks) ---")
db3 = SessionLocal()
with open("/tmp/whitespace_circular.txt", "w") as f:
    f.write("   \n\n\n   \t  \n")
try:
    circular_empty = ingest_circular(db3, "/tmp/whitespace_circular.txt", "Whitespace-Only Circular")
    check("Ingesting a whitespace-only file doesn't crash", True, f"circular_id={circular_empty.id}")
    assessments_empty = run_impact_assessment(db3, circular_empty)
    check("Running impact assessment on a zero-chunk circular doesn't crash", True, f"assessments={len(assessments_empty)}")
    check("Zero-chunk circular produces zero impact assessments (not a crash, not a false positive)",
          len(assessments_empty) == 0, f"count={len(assessments_empty)}")
except Exception as e:
    check("Whitespace-only circular handled gracefully", False, f"exception={type(e).__name__}: {e}")
db3.close()
os.remove("/tmp/whitespace_circular.txt")

os.remove("edge_test.db")
if os.path.exists("./data/edge_test_faiss2"):
    shutil.rmtree("./data/edge_test_faiss2")


print(f"\n{'='*50}")
print(f"EDGE CASE TEST RESULTS: {results['passed']} passed, {results['failed']} failed")
print(f"{'='*50}")

import sys as _sys
_sys.exit(1 if results["failed"] > 0 else 0)
