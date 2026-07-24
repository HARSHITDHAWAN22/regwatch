import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text
from app.db import Base


class Policy(Base):
    """
    An internal company policy/feature that circulars might impact.
    e.g. "KYC Verification Limit", "UPI Transaction Cap".
    This is the 'ground truth' the impact matcher checks new circulars against.
    """
    __tablename__ = "policies"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    owner_team = Column(String, nullable=True)
    based_on_circular_id = Column(String, nullable=True)  # what this policy currently complies with
    created_at = Column(DateTime, default=datetime.utcnow)
