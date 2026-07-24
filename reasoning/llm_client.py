"""
Factory for the LLM client + resilience wrapper.

- Factory pattern: one place to swap providers. Currently Google Gemini
  (genuinely free tier, no card required - see app/config.py comments).
  Swapping to OpenAI/Anthropic/local later means changing this one file,
  not touching any caller - that's the point of isolating it here.
- Retries transient failures with backoff (tenacity).
- Circuit breaker: after repeated failures, stop hammering the API and
  raise immediately so callers can fall back gracefully instead of hanging.
"""
import time
import json
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import get_settings

settings = get_settings()

_client: genai.Client | None = None


def get_llm_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class CircuitBreaker:
    """Simple circuit breaker: trips open after N consecutive failures,
    stays open for a cooldown period, then allows a trial request through."""

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.cooldown_seconds:
            self.opened_at = None
            self.failure_count = 0
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.opened_at = time.time()


_breaker = CircuitBreaker()


class LLMUnavailableError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_llm(system_prompt: str, user_prompt: str) -> tuple[str, int]:
    client = get_llm_client()
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",  # Gemini's equivalent of OpenAI's response_format=json_object
        ),
    )
    content = response.text
    # Gemini's usage_metadata gives token counts - fall back to 0 if unavailable
    tokens = 0
    if response.usage_metadata:
        tokens = (response.usage_metadata.prompt_token_count or 0) + \
                  (response.usage_metadata.candidates_token_count or 0)
    return content, tokens


def call_llm_json(system_prompt: str, user_prompt: str) -> tuple[dict, int, int]:
    """
    Returns (parsed_json_response, tokens_used, latency_ms).
    Raises LLMUnavailableError if the circuit breaker is open or all
    retries are exhausted - callers should catch this and use the
    rule-based fallback matcher instead of crashing.
    """
    if _breaker.is_open():
        raise LLMUnavailableError("Circuit breaker open - LLM temporarily unavailable")

    start = time.time()
    try:
        raw, tokens = _call_llm(system_prompt, user_prompt)
        _breaker.record_success()
        latency_ms = int((time.time() - start) * 1000)
        return json.loads(raw), tokens, latency_ms
    except Exception as e:
        print(f"REAL LLM CALL FAILED: {type(e).__name__}: {e}")
        _breaker.record_failure()
        raise LLMUnavailableError(str(e)) from e
