import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cache import cached_assessment, clear_cache

call_count = {"n": 0}


@cached_assessment
def fake_expensive_call(clause_text, policy_id, prompt_version):
    call_count["n"] += 1
    return {"impacts_policy": True, "reasoning": "fake"}


def test_cache_hit_avoids_recompute():
    clear_cache()
    call_count["n"] = 0
    r1 = fake_expensive_call("The transaction limit is Rs 1000", "policy1", "v1")
    r2 = fake_expensive_call("The transaction limit is Rs 1000", "policy1", "v1")
    assert call_count["n"] == 1          # only computed once
    assert r1["was_cache_hit"] is False
    assert r2["was_cache_hit"] is True


def test_different_keys_both_compute():
    clear_cache()
    call_count["n"] = 0
    fake_expensive_call("Clause about KYC limits", "policy1", "v1")
    fake_expensive_call("Clause about UPI caps", "policy1", "v1")
    assert call_count["n"] == 2


def test_same_text_different_chunk_ids_still_hits_cache():
    """The whole point of content-based keying: two DIFFERENT chunk records
    (e.g. from re-uploading the same circular) with IDENTICAL text should
    share a cache entry instead of re-costing an LLM call."""
    clear_cache()
    call_count["n"] = 0
    fake_expensive_call("Boilerplate legal clause text", "policy1", "v1")   # simulates chunk from upload #1
    fake_expensive_call("Boilerplate legal clause text", "policy1", "v1")   # simulates same text, chunk from upload #2
    assert call_count["n"] == 1
