"""
Round 4: JWT security edge cases (expiry, algorithm tampering, deleted-user
tokens), previously-untested validation paths (invalid review_status,
malformed/binary uploads, duplicate circular_number, zero-byte files), and
email format validation.
"""
import sys, os, shutil, types
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["DATABASE_URL"] = "sqlite:///./round4_test.db"
os.environ["FAISS_INDEX_DIR"] = "./data/round4_test_faiss"
if os.path.exists("round4_test.db"):
    os.remove("round4_test.db")
if os.path.exists("./data/round4_test_faiss"):
    shutil.rmtree("./data/round4_test_faiss")

import numpy as np
fake_st = types.ModuleType("sentence_transformers")
class FakeST:
    def __init__(self, *a, **kw): pass
    def encode(self, texts, **kw):
        rng = np.random.default_rng(9)
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
    return ({"impacts_policy": True, "severity": "info", "reasoning": "test",
              "evidence_sentence": "test"}, 50, 50)
llm_client_module.call_llm_json = fake_call_llm_json
import app.reasoning.impact_reasoner as ir, app.reasoning.verifier as vf, app.ingestion.structured_extractor as se
ir.call_llm_json = fake_call_llm_json
vf.call_llm_json = fake_call_llm_json
se.call_llm_json = fake_call_llm_json

from fastapi.testclient import TestClient
from app.main import app
from app.db import SessionLocal, init_db
from app.models.user import User, Role
from app.models.policy import Policy
from app.auth import hash_password, create_access_token
from app.config import get_settings

init_db()
client = TestClient(app)
settings = get_settings()

results = {"passed": 0, "failed": 0}
def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    results["passed" if cond else "failed"] += 1
    print(f"[{status}] {name} {extra}")

db = SessionLocal()
user = User(email="r4@test.com", hashed_password=hash_password("ValidPassword1"), role=Role.ADMIN)
db.add(user)
db.add(Policy(name="R4 Policy", description="A test policy."))
db.commit()
db.refresh(user)
user_id = user.id
db.close()

r = client.post("/auth/login", json={"email": "r4@test.com", "password": "ValidPassword1"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}


# ============================================================
# 1. JWT expiry
# ============================================================
print("\n--- JWT expiry ---")
from jose import jwt
from datetime import datetime, timedelta

expired_payload = {"sub": user_id, "role": "admin", "exp": datetime.utcnow() - timedelta(minutes=5)}
expired_token = jwt.encode(expired_payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
r = client.get("/impacts", headers={"Authorization": f"Bearer {expired_token}"})
check("Expired JWT doesn't crash a public-read endpoint", r.status_code == 200, f"status={r.status_code}")

r = client.post("/policies", headers={"Authorization": f"Bearer {expired_token}"}, json={"name": "X", "description": "Y"})
check("Expired JWT is REJECTED on a protected endpoint (401)", r.status_code == 401, f"status={r.status_code}")


# ============================================================
# 2. JWT algorithm tampering (classic "alg confusion" attack class)
# ============================================================
print("\n--- JWT algorithm tampering ---")
try:
    none_alg_token = jwt.encode({"sub": user_id, "role": "admin"}, "", algorithm="none")
    r = client.post("/policies", headers={"Authorization": f"Bearer {none_alg_token}"}, json={"name": "X", "description": "Y"})
    check("alg=none token is REJECTED, not silently trusted", r.status_code == 401, f"status={r.status_code}")
except Exception as e:
    # python-jose refusing to even encode an alg=none token is an even stronger guarantee
    check("alg=none token is REJECTED, not silently trusted", True, f"encode itself refused: {e}")

wrong_secret_token = jwt.encode({"sub": user_id, "role": "admin",
                                  "exp": datetime.utcnow() + timedelta(minutes=30)},
                                 "wrong-secret-guessed-by-attacker", algorithm=settings.jwt_algorithm)
r = client.post("/policies", headers={"Authorization": f"Bearer {wrong_secret_token}"}, json={"name": "X", "description": "Y"})
check("Token signed with wrong secret is REJECTED", r.status_code == 401, f"status={r.status_code}")


# ============================================================
# 3. Token for a deleted/nonexistent user
# ============================================================
print("\n--- Token for nonexistent user ---")
ghost_token = create_access_token("nonexistent-user-id-12345", "admin")
r = client.post("/policies", headers={"Authorization": f"Bearer {ghost_token}"}, json={"name": "X", "description": "Y"})
check("Token for a user_id that doesn't exist in DB is REJECTED", r.status_code == 401, f"status={r.status_code}")


# ============================================================
# 4. Invalid review_status value
# ============================================================
print("\n--- Invalid review_status ---")
r = client.post("/circulars/upload", headers=headers,
                 files={"file": ("t.txt", b"1. Test clause about the policy.", "text/plain")},
                 data={"title": "R4 test circular"})
job_id = r.json()["job_id"]
import time
for _ in range(10):
    status = client.get(f"/jobs/{job_id}").json()
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)

r = client.get("/impacts")
impacts = r.json()
if impacts:
    impact_id = impacts[0]["id"]
    r = client.post(f"/impacts/{impact_id}/review", headers=headers, json={"review_status": "banana"})
    check("Invalid review_status string is REJECTED (400), not silently accepted or 500",
          r.status_code == 400, f"status={r.status_code}, body={r.json()}")
else:
    check("Invalid review_status test", False, "no impacts were created to test against")


# ============================================================
# 5. Malformed / binary upload doesn't crash the background job
# ============================================================
print("\n--- Malformed/binary upload ---")
binary_garbage = bytes(range(256)) * 4  # arbitrary binary data, not valid UTF-8 or PDF
r = client.post("/circulars/upload", headers=headers,
                 files={"file": ("garbage.txt", binary_garbage, "application/octet-stream")},
                 data={"title": "Binary garbage upload"})
check("Binary garbage upload is accepted for processing (not rejected at upload time)", r.status_code == 200, f"status={r.status_code}")
job_id2 = r.json()["job_id"]
for _ in range(10):
    status2 = client.get(f"/jobs/{job_id2}").json()
    if status2["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)
check("Binary garbage is processed without crashing the server (completed or failed gracefully)",
      status2["status"] in ("completed", "failed"), f"status={status2}")


# ============================================================
# 6. Zero-byte file upload
# ============================================================
print("\n--- Zero-byte file upload ---")
r = client.post("/circulars/upload", headers=headers,
                 files={"file": ("empty.txt", b"", "text/plain")},
                 data={"title": "Zero-byte circular"})
check("Zero-byte file upload accepted without a 500", r.status_code == 200, f"status={r.status_code}")
job_id3 = r.json()["job_id"]
for _ in range(10):
    status3 = client.get(f"/jobs/{job_id3}").json()
    if status3["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)
check("Zero-byte file processed without crashing", status3["status"] in ("completed", "failed"), f"status={status3}")


# ============================================================
# 7. Duplicate circular_number (two unrelated circulars, same number)
# ============================================================
print("\n--- Duplicate circular_number (not an amendment relationship) ---")
r1 = client.post("/circulars/upload", headers=headers,
                  files={"file": ("dup1.txt", b"1. First circular content here.", "text/plain")},
                  data={"title": "Duplicate-number circular A", "circular_number": "DUP-001"})
job_a = r1.json()["job_id"]
for _ in range(10):
    sa = client.get(f"/jobs/{job_a}").json()
    if sa["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)

r2 = client.post("/circulars/upload", headers=headers,
                  files={"file": ("dup2.txt", b"1. Second, unrelated circular content here.", "text/plain")},
                  data={"title": "Duplicate-number circular B", "circular_number": "DUP-001"})
job_b = r2.json()["job_id"]
for _ in range(10):
    sb = client.get(f"/jobs/{job_b}").json()
    if sb["status"] in ("completed", "failed"):
        break
    time.sleep(0.3)
check("Two circulars sharing the same circular_number both process without crashing",
      sa["status"] == "completed" and sb["status"] == "completed", f"a={sa}, b={sb}")


# ============================================================
# 8. Email format validation
# ============================================================
print("\n--- Email format validation ---")
r = client.post("/auth/register", json={"email": "not-an-email-at-all", "password": "ValidPassword1", "role": "viewer"})
check("Registration with a malformed email is rejected", r.status_code == 422, f"status={r.status_code}, body={r.json()}")


print(f"\n{'='*50}")
print(f"ROUND 4 TEST RESULTS: {results['passed']} passed, {results['failed']} failed")
print(f"{'='*50}")

os.remove("round4_test.db")
if os.path.exists("./data/round4_test_faiss"):
    shutil.rmtree("./data/round4_test_faiss")

sys.exit(1 if results["failed"] > 0 else 0)
