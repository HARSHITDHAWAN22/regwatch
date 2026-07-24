"""
Core reasoning step: given a circular clause + a policy, ask the LLM
whether the clause impacts the policy, why, how severely, and which
exact sentence(s) justify that (span-level evidence for the audit trail).

Wrapped with @cached_assessment so identical (clause, policy, prompt_version)
triples never re-call the LLM. If the LLM is unavailable, falls back to a
rule-based keyword overlap heuristic and marks the result low_confidence
so it always gets routed to human review instead of silently failing.
"""
from app.cache import cached_assessment
from app.reasoning.llm_client import call_llm_json, LLMUnavailableError
from app.reasoning.feedback import format_few_shot_prompt
from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are a regulatory compliance analyst. You will be given a clause from a
regulatory circular and a description of an internal company policy. Decide whether the clause
impacts the policy.

A clause impacts a policy if ANY of the following are true:
- It directly changes a rule, limit, threshold, or requirement the policy governs.
- It imposes a NEW consequence, penalty, or liability tied to the policy's subject matter
  (e.g. "non-compliance with this timeline attracts a penalty" DOES impact a policy about
  that timeline, even though it doesn't restate the timeline itself).
- It uses different wording for the same underlying requirement the policy describes
  (e.g. a clause requiring "disclosure of APR before disbursement" impacts a policy titled
  "Loan Interest Rate Disclosure" even without using the word "disclosure requirement").
- It changes a related operational process (notification, authentication, verification, or
  reporting step) that the policy's description explicitly covers.

A clause does NOT impact a policy if it is a legal citation, boilerplate, an explicit exclusion
naming a DIFFERENT process/product than the policy governs, or genuinely unrelated in subject matter.

If past reviewer-validated examples are provided, use them to calibrate your judgment - especially
examples marked REJECTED, which show clauses that looked relevant but were confirmed by a human
reviewer to NOT actually impact the policy.

Respond ONLY with a JSON object in this exact shape:
{
  "impacts_policy": true or false,
  "severity": "info" | "action_required" | "urgent",
  "reasoning": "2-3 sentence explanation grounded ONLY in the clause text provided",
  "evidence_sentence": "the exact sentence from the clause that most directly justifies this assessment"
}

Rules:
- Do not require an exact wording match - judge substantive relevance to the policy's subject matter.
- When in doubt between a direct rule change and an indirect consequence, still flag impacts_policy
  as true if a compliance officer would reasonably need to review this clause for that policy.
- "urgent" severity means a deadline or immediate compliance action is implied.
- "evidence_sentence" MUST be copied verbatim from the clause text, not paraphrased.
- Never invent information not present in the clause."""


@cached_assessment
def assess_impact(clause_text: str, policy_id: str, prompt_version: str,
                   policy_name: str, policy_description: str, few_shot_examples: dict | None = None) -> dict:
    few_shot_block = format_few_shot_prompt(few_shot_examples) if few_shot_examples else ""

    user_prompt = f"""{few_shot_block}
CLAUSE:
{clause_text}

POLICY NAME: {policy_name}
POLICY DESCRIPTION: {policy_description}"""

    try:
        result, tokens, latency_ms = call_llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "impacts_policy": result.get("impacts_policy", False),
            "severity": result.get("severity", "info"),
            "reasoning": result.get("reasoning", ""),
            "evidence_sentence": result.get("evidence_sentence", ""),
            "tokens_used": tokens,
            "latency_ms": latency_ms,
            "method": "llm",
        }
    except LLMUnavailableError:
        return _fallback_assessment(clause_text, policy_name, policy_description)


def _fallback_assessment(clause_text: str, policy_name: str, policy_description: str) -> dict:
    """Rule-based degradation when the LLM is down - keeps the pipeline alive
    instead of crashing, at the cost of precision. Always flagged for review."""
    keywords = set(policy_name.lower().split()) | set(policy_description.lower().split())
    clause_words = set(clause_text.lower().split())
    overlap = keywords & clause_words

    return {
        "impacts_policy": len(overlap) >= 2,
        "severity": "action_required" if len(overlap) >= 2 else "info",
        "reasoning": f"[FALLBACK - LLM unavailable] Keyword overlap detected: {', '.join(overlap) or 'none'}",
        "evidence_sentence": clause_text[:200],
        "tokens_used": 0,
        "latency_ms": 0,
        "method": "fallback",
    }
