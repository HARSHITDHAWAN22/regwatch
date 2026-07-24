import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum
from app.db import Base


class Role(str, enum.Enum):
    ADMIN = "admin"        # manage policy registry
    REVIEWER = "reviewer"  # confirm/correct impact assessments
    VIEWER = "viewer"      # read-only


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(Role), default=Role.VIEWER)
    created_at = Column(DateTime, default=datetime.utcnow)
