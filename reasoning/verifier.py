"""
Verification / self-consistency pass. A second, independent LLM call
critiques the first assessment: "does this reasoning actually follow
from the cited clause?" This catches hallucinated reasoning that sounds
plausible but doesn't actually trace back to the source text.

Assessments scoring below settings.verification_min_score are flagged
is_flagged_for_review = True instead of being silently trusted.
"""
from app.reasoning.llm_client import call_llm_json, LLMUnavailableError
from app.config import get_settings

settings = get_settings()

CRITIC_SYSTEM_PROMPT = """You are a strict auditor reviewing another analyst's compliance assessment.
Given the original clause and the analyst's reasoning + evidence sentence, score how well the
reasoning is actually supported by the clause.

Respond ONLY with JSON:
{
  "score": 1-5,
  "critique": "one sentence explaining the score"
}

Scoring guide:
5 = evidence_sentence is verbatim from clause and reasoning follows directly from it
3 = reasoning is plausible but evidence_sentence is paraphrased or loosely connected
1 = reasoning does not follow from the clause, or evidence_sentence is not actually in the clause"""


def verify_assessment(clause_text: str, reasoning: str, evidence_sentence: str) -> dict:
    user_prompt = f"""CLAUSE:
{clause_text}

ANALYST REASONING: {reasoning}
ANALYST EVIDENCE SENTENCE: {evidence_sentence}"""

    try:
        result, tokens, latency_ms = call_llm_json(CRITIC_SYSTEM_PROMPT, user_prompt)
        score = int(result.get("score", 1))
        return {
            "score": score,
            "critique": result.get("critique", ""),
            "flagged_for_review": score < settings.verification_min_score,
            "tokens_used": tokens,
        }
    except LLMUnavailableError:
        # Can't verify -> be conservative and flag for human review
        return {
            "score": 0,
            "critique": "[FALLBACK] Verification unavailable - flagged for manual review",
            "flagged_for_review": True,
            "tokens_used": 0,
        }
