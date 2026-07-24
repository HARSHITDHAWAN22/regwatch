import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base


class CircularStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    AMENDED = "amended"
    SUPERSEDED = "superseded"


class Circular(Base):
    """
    A single regulatory circular (e.g. an RBI notification).
    supersedes_id links to an older circular this one amends/replaces -
    this is what powers the amendment graph + diff engine.
    """
    __tablename__ = "circulars"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    circular_number = Column(String, index=True, nullable=True)
    title = Column(String, nullable=False)
    source_filename = Column(String, nullable=True)
    raw_text = Column(Text, nullable=False)
    effective_date = Column(String, nullable=True)
    status = Column(Enum(CircularStatus), default=CircularStatus.ACTIVE)

    supersedes_id = Column(String, ForeignKey("circulars.id"), nullable=True)
    supersedes = relationship("Circular", remote_side=[id])

    structured_summary = Column(Text, nullable=True)  # JSON string: applies_to, action_required, penalty...
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("CircularChunk", back_populates="circular", cascade="all, delete-orphan")


class CircularChunk(Base):
    """Clause-level chunk of a circular - the unit that gets embedded/indexed."""
    __tablename__ = "circular_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    circular_id = Column(String, ForeignKey("circulars.id"), nullable=False)
    section_label = Column(String, nullable=True)   # e.g. "Para 4.2"
    text = Column(Text, nullable=False)
    char_start = Column(String, nullable=True)       # offsets in the original doc, for span highlighting
    char_end = Column(String, nullable=True)
    faiss_vector_id = Column(String, nullable=True)  # index position in FAISS

    circular = relationship("Circular", back_populates="chunks")
