"""
When circular B supersedes circular A, a compliance officer needs to know
*what specifically changed*, not just "there's a newer version." This asks
the LLM to produce a structured before/after diff grounded in both texts.

Same resilience pattern as the rest of the reasoning layer: falls back to
a plain "manual review needed" placeholder if the LLM is unavailable,
rather than crashing the ingestion pipeline over a non-critical feature.
"""
from app.reasoning.llm_client import call_llm_json, LLMUnavailableError

SYSTEM_PROMPT = """You are comparing an OLD regulatory circular against a NEW circular that
supersedes/amends it. Identify the concrete changes.

Respond ONLY with JSON:
{
  "changes": [
    {"aspect": "short label e.g. 'UPI Lite per-transaction limit'", "old_value": "...", "new_value": "..."}
  ],
  "summary": "one sentence overall summary of what changed"
}

If you cannot identify specific changes from the text provided, return an empty changes list
and explain why in the summary."""


def generate_diff(old_text: str, new_text: str) -> dict:
    user_prompt = f"""OLD CIRCULAR:
{old_text[:4000]}

NEW CIRCULAR:
{new_text[:4000]}"""

    try:
        result, tokens, latency_ms = call_llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "changes": result.get("changes", []),
            "summary": result.get("summary", ""),
            "tokens_used": tokens,
            "method": "llm",
        }
    except LLMUnavailableError:
        return {
            "changes": [],
            "summary": "[FALLBACK] Diff generation unavailable - manual comparison required.",
            "tokens_used": 0,
            "method": "fallback",
        }
