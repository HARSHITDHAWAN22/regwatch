import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base
from app.models.impact import ImpactAssessment, Severity, ReviewStatus
from app.reasoning.feedback import get_few_shot_examples, format_few_shot_prompt, few_shot_cache_signature


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _make_assessment(policy_id, review_status, cited_text, reasoning):
    return ImpactAssessment(
        circular_id="c1", circular_chunk_id="chunk1", policy_id=policy_id,
        cited_text=cited_text, reasoning=reasoning, severity=Severity.INFO,
        retrieval_score=0.9, review_status=review_status,
        prompt_version="v1", pipeline_version="v1",
    )


def test_no_feedback_yields_empty_prompt_block(db_session):
    examples = get_few_shot_examples(db_session, "policy-1")
    assert examples["positive"] == []
    assert examples["negative"] == []
    assert format_few_shot_prompt(examples) == ""


def test_confirmed_assessment_becomes_positive_example(db_session):
    db_session.add(_make_assessment("policy-1", ReviewStatus.CONFIRMED, "Clause X", "Reasoning X"))
    db_session.commit()

    examples = get_few_shot_examples(db_session, "policy-1")
    assert len(examples["positive"]) == 1
    assert examples["positive"][0]["clause"] == "Clause X"

    prompt_block = format_few_shot_prompt(examples)
    assert "CONFIRMED impact" in prompt_block
    assert "Clause X" in prompt_block


def test_rejected_assessment_becomes_negative_example(db_session):
    db_session.add(_make_assessment("policy-1", ReviewStatus.REJECTED, "Clause Y", "Reasoning Y"))
    db_session.commit()

    examples = get_few_shot_examples(db_session, "policy-1")
    assert len(examples["negative"]) == 1

    prompt_block = format_few_shot_prompt(examples)
    assert "REJECTED as false positive" in prompt_block
    assert "Clause Y" in prompt_block


def test_examples_scoped_to_correct_policy(db_session):
    db_session.add(_make_assessment("policy-1", ReviewStatus.CONFIRMED, "Clause A", "R-A"))
    db_session.add(_make_assessment("policy-2", ReviewStatus.CONFIRMED, "Clause B", "R-B"))
    db_session.commit()

    examples = get_few_shot_examples(db_session, "policy-1")
    assert len(examples["positive"]) == 1
    assert examples["positive"][0]["clause"] == "Clause A"


def test_pending_and_corrected_status_excluded_from_examples(db_session):
    """Only CONFIRMED/REJECTED are validated ground truth - PENDING hasn't
    been reviewed yet, and shouldn't be used to train future prompts."""
    db_session.add(_make_assessment("policy-1", ReviewStatus.PENDING, "Clause P", "R-P"))
    db_session.commit()

    examples = get_few_shot_examples(db_session, "policy-1")
    assert examples["positive"] == []
    assert examples["negative"] == []


def test_cache_signature_changes_when_feedback_changes(db_session):
    """This is the crux of why cache_key_extra exists: the signature MUST
    change when new feedback is added, or the LLM cache would keep serving
    a pre-feedback answer forever."""
    sig_before = few_shot_cache_signature(get_few_shot_examples(db_session, "policy-1"))

    db_session.add(_make_assessment("policy-1", ReviewStatus.REJECTED, "New clause", "New reasoning"))
    db_session.commit()

    sig_after = few_shot_cache_signature(get_few_shot_examples(db_session, "policy-1"))
    assert sig_before != sig_after


def test_example_count_capped(db_session):
    for i in range(5):
        db_session.add(_make_assessment("policy-1", ReviewStatus.CONFIRMED, f"Clause {i}", f"R-{i}"))
    db_session.commit()

    examples = get_few_shot_examples(db_session, "policy-1")
    assert len(examples["positive"]) == 2  # MAX_POSITIVE_EXAMPLES
