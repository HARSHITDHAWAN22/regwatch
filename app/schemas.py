from pydantic import BaseModel, field_validator, EmailStr
from datetime import datetime


class PolicyCreate(BaseModel):
    name: str
    description: str
    owner_team: str | None = None


class PolicyOut(BaseModel):
    id: str
    name: str
    description: str
    owner_team: str | None
    model_config = {"from_attributes": True}


class ImpactOut(BaseModel):
    id: str
    circular_id: str
    policy_id: str
    cited_text: str
    span_start: int | None
    span_end: int | None
    reasoning: str
    severity: str
    retrieval_score: float
    verification_score: int | None
    is_flagged_for_review: bool
    review_status: str
    created_at: datetime
    model_config = {"from_attributes": True}


class ReviewCorrection(BaseModel):
    review_status: str  # confirmed | corrected | rejected
    correction_note: str | None = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_within_bcrypt_limit(cls, v: str) -> str:
        # bcrypt silently truncates at 72 BYTES (not characters - multi-byte
        # UTF-8 chars make this smaller than 72 chars for non-ASCII input).
        # Silent truncation means two different passwords sharing the same
        # 72-byte prefix would both verify successfully against one hash -
        # rejecting overlong passwords explicitly is safer than letting that
        # happen invisibly.
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer (bcrypt limitation)")
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class RoleUpdate(BaseModel):
    role: str  # "admin" | "reviewer" | "viewer"