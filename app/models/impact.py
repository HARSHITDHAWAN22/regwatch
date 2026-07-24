import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Float, Enum, Integer, Boolean
from app.db import Base


class Severity(str, enum.Enum):
    INFO = "info"
    ACTION_REQUIRED = "action_required"
    URGENT = "urgent"


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class ImpactAssessment(Base):
    """
    The core output artifact + audit trail entry.
    Every row here is immutable once created (corrections create a linked
    feedback record, they don't overwrite this row) - that's what makes
    this defensible as a compliance audit log.
    """
    __tablename__ = "impact_assessments"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    circular_id = Column(String, nullable=False)
    circular_chunk_id = Column(String, nullable=False)
    policy_id = Column(String, nullable=False)

    cited_text = Column(Text, nullable=False)          # exact clause text
    span_start = Column(Integer, nullable=True)         # exact sentence offsets within cited_text
    span_end = Column(Integer, nullable=True)

    reasoning = Column(Text, nullable=False)             # LLM explanation
    severity = Column(Enum(Severity), nullable=False)
    retrieval_score = Column(Float, nullable=False)      # reranker/similarity score
    verification_score = Column(Integer, nullable=True)  # critic LLM 1-5 self-consistency score

    is_flagged_for_review = Column(Boolean, default=False)  # low verification score -> human review
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING)
    reviewer_correction = Column(Text, nullable=True)

    prompt_version = Column(String, nullable=False)
    pipeline_version = Column(String, nullable=False)

    # cost/latency observability
    tokens_used = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    was_cache_hit = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
