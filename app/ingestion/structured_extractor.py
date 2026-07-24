"""
Extracts structured metadata from a circular's raw text so the system
supports queries pure vector search can't ("show me circulars affecting
KYC with a penalty clause") and so amendment references can be detected.
"""
import json
from app.reasoning.llm_client import call_llm_json, LLMUnavailableError

SYSTEM_PROMPT = """Extract structured metadata from this regulatory circular text.
Respond ONLY with JSON in this exact shape:
{
  "applies_to": ["list of entity types this applies to, e.g. banks, NBFCs, payment aggregators"],
  "effective_date": "date string if mentioned, else null",
  "action_required": true or false,
  "penalty_mentioned": true or false,
  "references_other_circulars": ["any circular numbers/references mentioned as being amended, superseded, or referenced"]
}"""


def extract_structured_fields(circular_text: str) -> dict:
    # Truncate very long circulars for the extraction call - full text is still indexed/chunked separately
    excerpt = circular_text[:6000]
    try:
        result, _, _ = call_llm_json(SYSTEM_PROMPT, excerpt)
        return result
    except LLMUnavailableError:
        return {
            "applies_to": [],
            "effective_date": None,
            "action_required": None,
            "penalty_mentioned": None,
            "references_other_circulars": [],
            "extraction_failed": True,
        }
