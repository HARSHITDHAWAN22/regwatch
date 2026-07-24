import os
import shutil
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.circular import Circular
from app.models.policy import Policy
from app.models.impact import ImpactAssessment, ReviewStatus
from app.models.user import User, Role
from app.schemas import PolicyCreate, PolicyOut, ImpactOut, ReviewCorrection
from app.auth import get_current_user, require_role
from app.pipeline import ingest_circular, run_impact_assessment
from app.amendment.graph import find_superseding_circulars, find_amendment_chain
from app.cache import cache_stats
from app.job_store import set_job, update_job, get_job
from app.schemas import RoleUpdate

router = APIRouter(tags=["regwatch"])

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- Policy registry ----------

@router.post("/policies", response_model=PolicyOut)
def create_policy(payload: PolicyCreate, db: Session = Depends(get_db),
                   user: User = Depends(require_role(Role.ADMIN))):
    policy = Policy(name=payload.name, description=payload.description, owner_team=payload.owner_team)
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return policy


@router.get("/policies", response_model=list[PolicyOut])
def list_policies(db: Session = Depends(get_db)):
    return db.query(Policy).all()


# ---------- User management (admin only) ----------

@router.patch("/users/{user_id}/role")
def update_user_role(user_id: str, payload: RoleUpdate, db: Session = Depends(get_db),
                      user: User = Depends(require_role(Role.ADMIN))):
    """Only an existing admin can promote/demote another user's role.
    This is the ONLY way a user can become admin - never via self-registration."""
    if payload.role not in [r.value for r in Role]:
        raise HTTPException(status_code=400, detail="Invalid role")

    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    target.role = Role(payload.role)
    db.commit()
    return {"status": "updated", "user_id": user_id, "new_role": target.role.value}
    

# ---------- Circular ingestion (async via BackgroundTasks) ----------

def _process_circular_job(job_id: str, file_path: str, title: str, circular_number: str | None):
    from app.db import SessionLocal
    db = SessionLocal()
    try:
        update_job(job_id, status="processing")
        circular = ingest_circular(db, file_path, title, circular_number)
        update_job(job_id, circular_id=circular.id)

        assessments = run_impact_assessment(db, circular)
        update_job(job_id, status="completed", impacts_found=len(assessments))
    except Exception as e:
        update_job(job_id, status="failed", error=str(e))
    finally:
        db.close()


@router.post("/circulars/upload")
def upload_circular(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    circular_number: str | None = Form(None),
    file: UploadFile = File(...),
    user: User = Depends(require_role(Role.ADMIN, Role.REVIEWER)),
):
    """
    Returns immediately with a job_id - ingestion + LLM-based impact
    assessment happen in the background so uploads never block on slow
    LLM calls. Poll GET /jobs/{job_id} for status.
    """
    ext = os.path.splitext(file.filename)[1] or ".txt"
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())
    set_job(job_id, {"status": "queued"})
    background_tasks.add_task(_process_circular_job, job_id, saved_path, title, circular_number)
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/circulars")
def list_circulars(db: Session = Depends(get_db)):
    circulars = db.query(Circular).all()
    return [{"id": c.id, "title": c.title, "circular_number": c.circular_number,
              "status": c.status, "created_at": c.created_at} for c in circulars]


@router.get("/circulars/{circular_id}/amendments")
def get_amendment_info(circular_id: str, db: Session = Depends(get_db)):
    """Returns any newer circulars that supersede this one, plus the full
    chain of what this circular itself supersedes."""
    circular = db.query(Circular).filter(Circular.id == circular_id).first()
    if not circular:
        raise HTTPException(status_code=404, detail="Circular not found")
    return {
        "circular_id": circular_id,
        "superseded_by": find_superseding_circulars(db, circular_id),
        "supersedes_chain": find_amendment_chain(db, circular_id),
        "structured_summary": circular.structured_summary,
    }


# ---------- Impact assessments / audit log ----------

@router.get("/impacts", response_model=list[ImpactOut])
def list_impacts(severity: str | None = None, flagged_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(ImpactAssessment)
    if severity:
        query = query.filter(ImpactAssessment.severity == severity)
    if flagged_only:
        query = query.filter(ImpactAssessment.is_flagged_for_review == True)  # noqa: E712
    return query.order_by(ImpactAssessment.created_at.desc()).all()


@router.post("/impacts/{impact_id}/review")
def review_impact(impact_id: str, payload: ReviewCorrection, db: Session = Depends(get_db),
                   user: User = Depends(require_role(Role.ADMIN, Role.REVIEWER))):
    """
    Human feedback loop: a reviewer confirms/corrects/rejects an assessment.
    The original row is never overwritten (audit-log immutability) - we only
    update review_status + attach the correction note.
    """
    assessment = db.query(ImpactAssessment).filter(ImpactAssessment.id == impact_id).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    if payload.review_status not in [s.value for s in ReviewStatus]:
        raise HTTPException(status_code=400, detail="Invalid review_status")

    assessment.review_status = ReviewStatus(payload.review_status)
    assessment.reviewer_correction = payload.correction_note
    db.commit()
    return {"status": "updated", "impact_id": impact_id}


# ---------- Observability ----------

@router.get("/metrics/by-version")
def get_metrics_by_version(db: Session = Depends(get_db)):
    """
    Groups impact assessments by prompt_version/pipeline_version so you can
    compare outcomes before/after a prompt change - e.g. 'did flagging rate
    or average verification score improve after v2?' - instead of only ever
    seeing an aggregate that hides regressions introduced by a prompt edit.
    """
    from sqlalchemy import func as sql_func, cast, Integer

    rows = (
        db.query(
            ImpactAssessment.prompt_version,
            ImpactAssessment.pipeline_version,
            sql_func.count(ImpactAssessment.id).label("total"),
            sql_func.sum(cast(ImpactAssessment.is_flagged_for_review, Integer)).label("flagged"),
            sql_func.avg(ImpactAssessment.verification_score).label("avg_verification_score"),
            sql_func.sum(ImpactAssessment.tokens_used).label("total_tokens"),
        )
        .group_by(ImpactAssessment.prompt_version, ImpactAssessment.pipeline_version)
        .all()
    )

    return [
        {
            "prompt_version": r.prompt_version,
            "pipeline_version": r.pipeline_version,
            "total_assessments": r.total,
            "flagged_for_review": int(r.flagged or 0),
            "avg_verification_score": round(r.avg_verification_score, 2) if r.avg_verification_score else None,
            "total_tokens": r.total_tokens or 0,
        }
        for r in rows
    ]


@router.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    total = db.query(ImpactAssessment).count()
    flagged = db.query(ImpactAssessment).filter(ImpactAssessment.is_flagged_for_review == True).count()  # noqa: E712
    total_tokens = db.query(ImpactAssessment).with_entities(ImpactAssessment.tokens_used).all()
    tokens_sum = sum(t[0] or 0 for t in total_tokens)

    return {
        "total_assessments": total,
        "flagged_for_review": flagged,
        "flagged_rate": round(flagged / total, 3) if total else 0,
        "total_tokens_used": tokens_sum,
        "estimated_cost_usd": round(tokens_sum / 1_000_000 * 0.15, 4),  # gpt-4o-mini approx rate
        "cache": cache_stats(),
    }
