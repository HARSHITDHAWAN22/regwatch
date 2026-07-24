"""
Closes the human feedback loop: past reviewer decisions (confirmed /
rejected) for a given policy are pulled and formatted as few-shot examples,
which get injected into the impact-reasoning prompt for future assessments
against that same policy. This is what makes review effort compound instead
of disappearing into an audit log nobody re-uses.

- CONFIRMED assessments -> positive examples ("this kind of clause DOES impact this policy")
- REJECTED assessments -> negative examples ("this looked similar but does NOT impact this policy" - directly improves precision by teaching the model to avoid a specific false-positive pattern it already made once)

Capped at a small number of each so the prompt doesn't grow unbounded as
the feedback history accumulates.
"""
from sqlalchemy.orm import Session
from app.models.impact import ImpactAssessment, ReviewStatus

MAX_POSITIVE_EXAMPLES = 2
MAX_NEGATIVE_EXAMPLES = 2


def get_few_shot_examples(db: Session, policy_id: str) -> dict:
    confirmed = (
        db.query(ImpactAssessment)
        .filter(ImpactAssessment.policy_id == policy_id, ImpactAssessment.review_status == ReviewStatus.CONFIRMED)
        .order_by(ImpactAssessment.created_at.desc())
        .limit(MAX_POSITIVE_EXAMPLES)
        .all()
    )
    rejected = (
        db.query(ImpactAssessment)
        .filter(ImpactAssessment.policy_id == policy_id, ImpactAssessment.review_status == ReviewStatus.REJECTED)
        .order_by(ImpactAssessment.created_at.desc())
        .limit(MAX_NEGATIVE_EXAMPLES)
        .all()
    )

    return {
        "positive": [{"clause": c.cited_text, "reasoning": c.reasoning} for c in confirmed],
        "negative": [{"clause": r.cited_text, "reasoning": r.reasoning} for r in rejected],
    }


def format_few_shot_prompt(examples: dict) -> str:
    """Renders the examples dict into the text block injected into the LLM prompt.
    Returns empty string if there's nothing to inject yet (cold start / no feedback)."""
    if not examples["positive"] and not examples["negative"]:
        return ""

    lines = ["PAST REVIEWER-VALIDATED EXAMPLES FOR THIS POLICY (learn from these):"]
    for ex in examples["positive"]:
        lines.append(f'- CONFIRMED impact: "{ex["clause"][:200]}" -> {ex["reasoning"][:150]}')
    for ex in examples["negative"]:
        lines.append(f'- REJECTED as false positive: "{ex["clause"][:200]}" -> do NOT flag similar clauses as impacting this policy')
    return "\n".join(lines) + "\n"


def few_shot_cache_signature(examples: dict) -> str:
    """A short, stable signature of the current example set - used so the
    LLM-response cache correctly invalidates when new feedback changes what
    would be injected, instead of silently serving a stale pre-feedback answer."""
    import hashlib
    raw = str(sorted(e["clause"] for e in examples["positive"])) + str(sorted(e["clause"] for e in examples["negative"]))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
