"""
Orchestrates the full RegWatch pipeline (Chain of Responsibility):

  ingest -> extract structure -> chunk -> index (dense+sparse)
  -> for each policy: hybrid retrieve -> rerank -> reason -> verify -> persist

This is the one place that knows the whole flow; every stage is a small,
independently-testable function imported from elsewhere.
"""
import json
from sqlalchemy.orm import Session
from app.config import get_settings
from app.models.circular import Circular, CircularChunk, CircularStatus
from app.models.policy import Policy
from app.models.impact import ImpactAssessment, Severity, ReviewStatus
from app.ingestion.pdf_parser import parse_document
from app.ingestion.chunker import get_chunker
from app.ingestion.structured_extractor import extract_structured_fields
from app.embeddings.embedder import embed_texts
from app.vectorstore.faiss_store import get_faiss_store
from app.vectorstore.bm25_store import get_bm25_store
from app.vectorstore.hybrid_retriever import hybrid_search
from app.vectorstore.reranker import rerank
from app.reasoning.impact_reasoner import assess_impact
from app.reasoning.verifier import verify_assessment
from app.reasoning.feedback import get_few_shot_examples, few_shot_cache_signature
from app.amendment.reference_detector import link_amendment
from app.amendment.diff_engine import generate_diff
from app.alerts import maybe_alert_on_impact

settings = get_settings()


def ingest_circular(db: Session, file_path: str, title: str, circular_number: str | None = None) -> Circular:
    """Stage 1-2: parse, extract structure, chunk, index. Returns the persisted Circular."""
    raw_text = parse_document(file_path)
    structured = extract_structured_fields(raw_text)

    circular = Circular(
        title=title,
        circular_number=circular_number,
        source_filename=file_path,
        raw_text=raw_text,
        status=CircularStatus.ACTIVE,
        structured_summary=json.dumps(structured),
    )
    db.add(circular)
    db.flush()  # get circular.id without committing yet

    chunker = get_chunker("section")
    chunks = chunker.chunk(raw_text)
    if not chunks:
        db.commit()
        return circular

    chunk_texts = [c.text for c in chunks]
    vectors = embed_texts(chunk_texts)

    chunk_records = []
    for chunk in chunks:
        record = CircularChunk(
            circular_id=circular.id,
            section_label=chunk.section_label,
            text=chunk.text,
            char_start=str(chunk.char_start),
            char_end=str(chunk.char_end),
        )
        db.add(record)
        chunk_records.append(record)
    db.flush()

    chunk_ids = [r.id for r in chunk_records]
    get_faiss_store().add(vectors, chunk_ids)
    get_bm25_store().add(chunk_ids, chunk_texts)

    db.commit()
    db.refresh(circular)

    # Amendment detection: does this circular reference/supersede an existing one?
    referenced = structured.get("references_other_circulars", []) or []
    superseded = link_amendment(db, circular, referenced)
    if superseded:
        diff = generate_diff(superseded.raw_text, circular.raw_text)
        circular.structured_summary = json.dumps({**structured, "amendment_diff": diff})
        db.add(circular)
        db.commit()
        db.refresh(circular)

    return circular


def run_impact_assessment(db: Session, circular: Circular) -> list[ImpactAssessment]:
    """
    Stage 3-6: for every policy, hybrid-retrieve relevant clauses from this
    circular, rerank, reason, verify, and persist an audit-logged assessment
    for every clause-policy pair that clears the impact threshold.
    """
    policies = db.query(Policy).all()
    chunk_lookup = {c.id: c for c in circular.chunks}
    created: list[ImpactAssessment] = []

    for policy in policies:
        query = f"{policy.name}: {policy.description}"

        # Pull reviewer feedback for this policy once per run - this is the
        # feedback loop actually closing: past confirm/reject decisions
        # shape how future clauses against this same policy get assessed.
        few_shot = get_few_shot_examples(db, policy.id)
        few_shot_sig = few_shot_cache_signature(few_shot)

        fused = hybrid_search(query, top_k=15)
        # keep only chunks that belong to THIS circular
        candidates = [
            (cid, chunk_lookup[cid].text)
            for cid, _ in fused
            if cid in chunk_lookup
        ]
        if not candidates:
            continue

        reranked = rerank(query, candidates, top_k=settings.rerank_top_k)

        for chunk_id, chunk_text, rerank_score in reranked:
            if rerank_score < settings.similarity_threshold:
                continue

            result = assess_impact(
                chunk_text, policy.id, settings.prompt_version,
                policy_name=policy.name, policy_description=policy.description,
                few_shot_examples=few_shot, cache_key_extra=few_shot_sig,
            )

            if not result.get("impacts_policy"):
                continue

            verification = verify_assessment(chunk_text, result["reasoning"], result["evidence_sentence"])

            span_start = chunk_text.find(result["evidence_sentence"])
            span_end = span_start + len(result["evidence_sentence"]) if span_start >= 0 else None

            assessment = ImpactAssessment(
                circular_id=circular.id,
                circular_chunk_id=chunk_id,
                policy_id=policy.id,
                cited_text=chunk_text,
                span_start=span_start if span_start >= 0 else None,
                span_end=span_end,
                reasoning=result["reasoning"],
                severity=Severity(result.get("severity", "info")),
                retrieval_score=rerank_score,
                verification_score=verification["score"],
                is_flagged_for_review=verification["flagged_for_review"] or result.get("method") == "fallback",
                review_status=ReviewStatus.PENDING,
                prompt_version=settings.prompt_version,
                pipeline_version=settings.pipeline_version,
                tokens_used=result.get("tokens_used", 0) + verification.get("tokens_used", 0),
                latency_ms=result.get("latency_ms", 0),
                was_cache_hit=result.get("was_cache_hit", False),
            )
            db.add(assessment)
            created.append(assessment)

            maybe_alert_on_impact(
                policy_name=policy.name,
                severity=assessment.severity.value,
                circular_title=circular.title,
                reasoning=assessment.reasoning,
            )

    db.commit()
    return created
