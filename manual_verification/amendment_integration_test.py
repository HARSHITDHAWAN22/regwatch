import sys, os, shutil, time, types
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

os.environ["DATABASE_URL"] = "sqlite:///./test_amendment.db"
os.environ["FAISS_INDEX_DIR"] = "./data/test_amendment_faiss"
if os.path.exists("test_amendment.db"):
    os.remove("test_amendment.db")
if os.path.exists("./data/test_amendment_faiss"):
    shutil.rmtree("./data/test_amendment_faiss")

import numpy as np

fake_st = types.ModuleType("sentence_transformers")
class FakeST:
    def __init__(self, *a, **kw): pass
    def encode(self, texts, **kw):
        rng = np.random.default_rng(42)
        return np.array([rng.random(384).astype("float32") for _ in texts])
class FakeCE:
    def __init__(self, *a, **kw): pass
    def predict(self, pairs):
        # Deterministic keyword-overlap heuristic instead of random scores -
        # unseeded np.random here previously caused flaky test runs where
        # scores randomly fell below the similarity threshold.
        scores = []
        for query, text in pairs:
            overlap = len(set(query.lower().split()) & set(text.lower().split()))
            scores.append(overlap / 10.0 + 0.5)
        return np.array(scores)
fake_st.SentenceTransformer = FakeST
fake_st.CrossEncoder = FakeCE
sys.modules["sentence_transformers"] = fake_st

import app.reasoning.llm_client as llm_client_module

def fake_call_llm_json(system_prompt, user_prompt):
    if "applies_to" in system_prompt:
        # first circular: no references. second circular: references SAMPLE-01
        refs = ["SAMPLE-01"] if "AMENDED VERSION" in user_prompt else []
        return ({"applies_to": ["banks"], "effective_date": "2025-01-31",
                  "action_required": True, "penalty_mentioned": True,
                  "references_other_circulars": refs}, 100, 200)
    if "comparing an OLD" in system_prompt:
        return ({"changes": [{"aspect": "UPI Lite per-transaction limit",
                               "old_value": "Rs. 1,000", "new_value": "Rs. 2,000"}],
                  "summary": "Per-transaction limit doubled."}, 120, 250)
    if "strict auditor" in system_prompt:
        return ({"score": 5, "critique": "Well grounded."}, 50, 100)
    return ({"impacts_policy": True, "severity": "urgent",
              "reasoning": "This clause raises the UPI Lite limit again, requiring urgent system changes.",
              "evidence_sentence": "limit shall be further enhanced"}, 150, 300)

llm_client_module.call_llm_json = fake_call_llm_json
import app.reasoning.impact_reasoner as ir
import app.reasoning.verifier as vf
import app.ingestion.structured_extractor as se
import app.amendment.diff_engine as de
ir.call_llm_json = fake_call_llm_json
vf.call_llm_json = fake_call_llm_json
se.call_llm_json = fake_call_llm_json
de.call_llm_json = fake_call_llm_json

from app.db import SessionLocal, init_db
from app.models.policy import Policy
from app.pipeline import ingest_circular, run_impact_assessment
from app.amendment.graph import find_superseding_circulars, find_amendment_chain
from app.alerts import subscribe, AlertChannel

init_db()

captured_alerts = []
class CapturingChannel(AlertChannel):
    def notify(self, message, severity, context):
        captured_alerts.append((message, severity, context))
subscribe(CapturingChannel())

db = SessionLocal()
db.add(Policy(name="UPI Transaction Cap", description="UPI transaction limits."))
db.commit()

# Ingest original circular
os.makedirs("/tmp/amendtest", exist_ok=True)
with open("/tmp/amendtest/circular1.txt", "w") as f:
    f.write("1. Per-transaction limit for UPI Lite shall be enhanced from Rs. 500 to Rs. 1,000.")
c1 = ingest_circular(db, "/tmp/amendtest/circular1.txt", "Original UPI Circular", "SAMPLE-01")
run_impact_assessment(db, c1)

# Ingest amending circular referencing the first
with open("/tmp/amendtest/circular2.txt", "w") as f:
    f.write("1. AMENDED VERSION: The UPI Lite per-transaction limit shall be further enhanced to Rs. 2,000.")
c2 = ingest_circular(db, "/tmp/amendtest/circular2.txt", "Amending UPI Circular", "SAMPLE-02")
run_impact_assessment(db, c2)

results = {"passed": 0, "failed": 0}
def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    results["passed" if cond else "failed"] += 1
    print(f"[{status}] {name} {extra}")

# Check amendment link was made
db.refresh(c2)
check("New circular's supersedes_id points to original", c2.supersedes_id == c1.id,
      f"supersedes_id={c2.supersedes_id}, c1.id={c1.id}")

check("Diff was generated and stored in structured_summary",
      "amendment_diff" in (c2.structured_summary or ""),
      f"summary={c2.structured_summary[:200]}")

import json as _json
try:
    parsed_summary = _json.loads(c2.structured_summary)
    check("structured_summary is valid, parseable JSON (not a Python repr string)", True,
          f"parsed keys={list(parsed_summary.keys())}")
    check("Parsed JSON contains the amendment_diff with real changes", 
          len(parsed_summary.get("amendment_diff", {}).get("changes", [])) > 0,
          f"changes={parsed_summary.get('amendment_diff', {}).get('changes')}")
except _json.JSONDecodeError as e:
    check("structured_summary is valid, parseable JSON (not a Python repr string)", False, f"JSONDecodeError: {e}")

superseding = find_superseding_circulars(db, c1.id)
check("Amendment graph finds superseding circular for c1", len(superseding) == 1 and superseding[0]["id"] == c2.id,
      f"found={superseding}")

chain = find_amendment_chain(db, c2.id)
check("Amendment chain from c2 leads back to c1", len(chain) == 1 and chain[0]["id"] == c1.id,
      f"chain={chain}")

check("Urgent-severity alert was fired", len(captured_alerts) > 0, f"alerts={captured_alerts}")
if captured_alerts:
    check("Alert references correct policy", "UPI Transaction Cap" in captured_alerts[0][2].get("policy", ""),
          f"context={captured_alerts[0][2]}")

db.close()
print(f"\n{'='*50}\nAMENDMENT/ALERTING TEST: {results['passed']} passed, {results['failed']} failed\n{'='*50}")

os.remove("test_amendment.db")
shutil.rmtree("/tmp/amendtest")
sys.exit(1 if results["failed"] > 0 else 0)
