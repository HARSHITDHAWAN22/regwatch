"""
Run once after first startup to populate the policy registry and create
a demo admin user:

    python -m scripts.seed
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import SessionLocal, init_db
from app.models.policy import Policy
from app.models.user import User, Role
from app.auth import hash_password


def seed_policies(db):
    with open("data/seed_policies.json") as f:
        policies = json.load(f)

    existing = {p.name for p in db.query(Policy).all()}
    added = 0
    for p in policies:
        if p["name"] in existing:
            continue
        db.add(Policy(name=p["name"], description=p["description"], owner_team=p.get("owner_team")))
        added += 1
    db.commit()
    print(f"Seeded {added} new policies ({len(existing)} already existed).")


def seed_admin_user(db):
    if db.query(User).filter(User.email == "admin@regwatch.demo").first():
        print("Admin user already exists.")
        return
    user = User(
        email="admin@regwatch.demo",
        hashed_password=hash_password("admin123"),
        role=Role.ADMIN,
    )
    db.add(user)
    db.commit()
    print("Created demo admin user: admin@regwatch.demo / admin123  (CHANGE THIS IN PRODUCTION)")


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        seed_policies(db)
        seed_admin_user(db)
    finally:
        db.close()
