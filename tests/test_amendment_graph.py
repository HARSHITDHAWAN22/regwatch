import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base
from app.models.circular import Circular, CircularStatus
from app.amendment.graph import find_superseding_circulars, find_amendment_chain


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_find_superseding_circulars(db_session):
    old = Circular(title="Old Rule", circular_number="C-1", raw_text="old text", status=CircularStatus.AMENDED)
    db_session.add(old)
    db_session.flush()

    new = Circular(title="New Rule", circular_number="C-2", raw_text="new text",
                    supersedes_id=old.id, status=CircularStatus.ACTIVE)
    db_session.add(new)
    db_session.commit()

    result = find_superseding_circulars(db_session, old.id)
    assert len(result) == 1
    assert result[0]["id"] == new.id


def test_find_superseding_circulars_none_when_not_amended(db_session):
    standalone = Circular(title="Standalone Rule", circular_number="C-3", raw_text="text", status=CircularStatus.ACTIVE)
    db_session.add(standalone)
    db_session.commit()

    result = find_superseding_circulars(db_session, standalone.id)
    assert result == []


def test_amendment_chain_multi_hop(db_session):
    """C-3 supersedes C-2 supersedes C-1 - chain from C-3 should reach both."""
    c1 = Circular(title="v1", circular_number="C-1", raw_text="t1", status=CircularStatus.SUPERSEDED)
    db_session.add(c1)
    db_session.flush()

    c2 = Circular(title="v2", circular_number="C-2", raw_text="t2", supersedes_id=c1.id, status=CircularStatus.AMENDED)
    db_session.add(c2)
    db_session.flush()

    c3 = Circular(title="v3", circular_number="C-3", raw_text="t3", supersedes_id=c2.id, status=CircularStatus.ACTIVE)
    db_session.add(c3)
    db_session.commit()

    chain = find_amendment_chain(db_session, c3.id)
    chain_ids = [c["id"] for c in chain]
    assert chain_ids == [c2.id, c1.id]
