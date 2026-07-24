"""
Detects when a newly-ingested circular references/supersedes an earlier
one already in the DB. Uses the structured_extractor's
references_other_circulars field (LLM-extracted) matched against existing
circular_number values - a simple, robust approach that doesn't require
brittle regex parsing of every possible citation format regulators use.
"""
from sqlalchemy.orm import Session
from app.models.circular import Circular


def link_amendment(db: Session, new_circular: Circular, referenced_numbers: list[str]) -> Circular | None:
    """
    If any referenced circular_number matches an existing circular in the
    DB, set new_circular.supersedes_id to point at it and mark the old one
    AMENDED. Returns the superseded Circular if a link was made, else None.
    """
    from app.models.circular import CircularStatus

    for ref in referenced_numbers:
        old = db.query(Circular).filter(
            Circular.circular_number == ref,
            Circular.id != new_circular.id,
        ).first()
        if old:
            new_circular.supersedes_id = old.id
            old.status = CircularStatus.AMENDED
            db.add(old)
            db.add(new_circular)
            db.commit()
            db.refresh(new_circular)
            return old
    return None
