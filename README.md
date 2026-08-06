# RegWatch

Most compliance teams still handle regulatory impact analysis the same way they did fifteen years ago: someone reads a new circular, then manually cross-references it against a spreadsheet of internal policies to figure out what needs to change. RegWatch automates that first pass — it ingests a circular, figures out which policies it actually touches, and hands a reviewer a citable, evidence-backed finding instead of a blank page.

[Live demo](https://regwatch-jubsbzgcdgfkresoytq94f.streamlit.app/) · [Repo](https://github.com/HARSHITDHAWAN22/regwatch)

## The pipeline

A circular gets split into clauses. Each clause is checked against the policy registry using two retrieval methods running in parallel — dense embedding search (FAISS) and keyword search (BM25) — merged with Reciprocal Rank Fusion so a clause doesn't get missed just because it's semantically vague but keyword-precise, or vice versa. A cross-encoder reranks the merged candidates, and Gemini makes the actual call on whether the clause impacts each policy — instructed to abstain rather than guess, and required to cite the exact sentence that justifies its answer. A second, independent LLM pass then checks that first judgment before a human ever sees it.

Reviewers confirm, reject, or correct findings, and every correction becomes a few-shot example for future calls. The system is meant to get sharper on the specific mistakes your reviewers actually catch, not stay static after launch.

If Gemini starts failing repeatedly — rate limits, outages, whatever — a circuit breaker trips and the pipeline falls back to a much dumber keyword-overlap heuristic instead of hanging. It's a worse answer, but the system stays up.

## What's actually verified, not just claimed

I ran a real evaluation — 10 hand-labeled clause/policy pairs, live Gemini calls, no mocking:

| Precision | Recall | F1 |
|---|---|---|
| 80% | 80% | 80% |

That's a small sample and I know it. A 30-case expanded set exists in the repo (`eval/eval_set_expanded.json`), but I haven't run it cleanly against a full daily API quota yet — the free tier caps at 20 calls/day, and I'd rather report an honest small number than a big one I can't stand behind.

**Test coverage:** 27 unit tests, plus 98 checks spread across six manual verification scripts — integration, amendment tracking, edge cases, concurrency, a dedicated security pass, and one test that genuinely kills the Redis process mid-run to confirm the fallback actually works instead of just assuming it does. All 125 currently pass, and GitHub Actions runs the unit suite on every push — check the repo's Actions tab for the live status.

**A real security bug, found post-deployment:** self-registration originally let a caller pick their own role, including `admin`, with nothing checking it — anyone could `POST /auth/register` with `role: admin` and walk away with an admin token. I caught this, fixed self-registration to always default to `viewer`, and added a separate role-gated endpoint that only an existing admin can use to promote someone. Verified live: a registration request asking for `admin` now correctly comes back `viewer`, and an admin-only action with that token correctly 403s.

**Other bugs worth mentioning, because they weren't the obvious kind:** the response cache was keyed on a random per-upload UUID instead of the clause text itself, so it silently never hit on repeat uploads — nothing errored, it just quietly did nothing useful. The FAISS/BM25/cache singletons had zero locking despite being written to from concurrent async tasks. And bcrypt truncates passwords past 72 bytes by default, which means two different long passwords can verify against the same hash unless you check for it — I added that check at registration instead of finding out the hard way later.

## Stack

FastAPI · SQLAlchemy · FAISS · BM25 · sentence-transformers cross-encoder · Gemini · Redis (in-memory fallback if Redis isn't reachable) · JWT auth with role-based access · Streamlit · Docker · deployed on Render + Streamlit Community Cloud · GitHub Actions

## Honest limitations

The live demo runs on free-tier hosting — 512MB RAM cap. Login, the policy registry, and metrics are rock solid there; circular processing itself can occasionally exceed that limit under load and restart the backend mid-request. Doesn't happen locally, where there's no such ceiling.

RBAC is role-level (admin/reviewer/viewer), not resource-level — there's no "you can only touch what you uploaded" yet. And the eval set is small, as covered above.

Feedback quality is also fully trusted — a reviewer's confirm/reject decision gets fed directly into future LLM prompts as a few-shot example, with no check on whether that feedback was actually correct. A malicious or compromised reviewer account could repeatedly submit bad feedback (rejecting real impacts, confirming false ones) and gradually bias the model's judgment for that policy, since there's currently no rate limiting or anomaly detection on review actions.

## Future improvements

A few things I'd change once this needs to handle more than a demo's worth of load:

- **Postgres instead of SQLite** — SQLite locks the whole file on every write, which is fine solo but won't hold up with multiple reviewers writing at once. Already just a `DATABASE_URL` change away, no rewrite needed.

- **Elasticsearch instead of in-memory BM25** — the current keyword index rebuilds itself from scratch on every add, which is a non-issue at a few thousand chunks but would start to hurt on a much bigger circular archive.

- **A paid LLM tier (or a different provider)** — the free Gemini tier caps at 20 calls/day, which is the actual reason the full 30-case eval set hasn't been run cleanly yet. Swapping providers is a one-file change thanks to the factory pattern in `llm_client.py`.

- **Resource-level RBAC** — right now any reviewer can review any circular. Locking that down to "only what you uploaded or were assigned" just needs an extra ownership field and a permission check next to the existing role check.

- **Celery/RQ instead of BackgroundTasks** — current background jobs have no retry logic and run in a single process. Worth doing once retries or multi-worker scaling actually matter.

- **A real frontend (React/Next.js)** — Streamlit got a working reviewer UI out fast, which is what I needed. A production version with real users would eventually want something more polished.

- **Rate limiting on review actions** — a reviewer can currently confirm/reject as many findings as they want, as fast as they want. Capping how many reviews one account can submit in a given window would limit how much damage a single compromised or malicious account could do before being noticed.

- **Suspicious feedback pattern detection** — no current mechanism flags a reviewer whose confirm/reject behavior looks off (e.g. rejecting far more than other reviewers, or flip-flopping on the same policy repeatedly). Tracking per-reviewer patterns and flagging outliers to an admin would catch feedback-poisoning early instead of letting it silently bias future LLM judgments.

## Running it locally

Python 3.11. Redis is optional — falls back to in-memory automatically if nothing's running.

```bash
git clone https://github.com/HARSHITDHAWAN22/regwatch.git
cd regwatch

python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows

pip install -r requirements.txt

cp .env.example .env
# set GEMINI_API_KEY — free key at aistudio.google.com/apikey

python -m scripts.seed          # creates admin@regwatch.demo / admin123
```

Two terminals:

```bash
uvicorn app.main:app --reload
```
```bash
streamlit run frontend/streamlit_app.py
```

Open `localhost:8501`, log in, upload one of the sample circulars in `data/sample_circulars/`, and watch the whole thing run end to end.

Or skip the venv entirely: `docker-compose up`

**Tests:**
```bash
pytest tests/ -v
python manual_verification/integration_test.py
python manual_verification/edge_case_test.py
python manual_verification/concurrency_test.py
python manual_verification/round4_security_test.py
python manual_verification/redis_concurrent_http_test.py
python manual_verification/redis_fallback_test.py
python -m eval.evaluate
```

## License

MIT
