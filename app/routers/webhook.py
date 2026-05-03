"""Webhook endpoints for the Salesforce → IDP audit pipeline.

Two endpoints:
  POST /webhook/pdf-attached   — Salesforce signals PDF attached to case.
                                 IDP starts extraction in background.
  POST /webhook/mulesoft-done  — Salesforce signals MuleSoft completed.
                                 IDP runs comparison + builds findings.

Plus a read endpoint:
  GET  /audit/cases/{case_id}  — Inspect job state + findings text.

Each webhook returns 202 Accepted immediately and runs the heavy work in
a background task. Idempotency: repeat webhooks for the same case_id are
de-duplicated via JobStore state.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.dependencies import get_settings
from app.services.audit.job_store import get_job_store
from app.services.audit.jobs import run_comparison, run_extraction

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _verify_webhook_mode(settings: Settings = Depends(get_settings)) -> None:
    """Reject webhook requests when audit_mode is set to 'poll' only."""
    if settings.audit_mode == "poll":
        raise HTTPException(
            status_code=503,
            detail=(
                "IDP is configured for polling mode (IDP_AUDIT_MODE=poll). "
                "Webhooks are disabled. Set IDP_AUDIT_MODE=webhook or =both "
                "to enable webhook reception."
            ),
        )


def _verify_webhook_token(
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
) -> None:
    """Reject webhooks lacking a matching bearer token.

    Auth is REQUIRED unless IDP_DEV_MODE=true is explicitly set. Missing
    token in production fails closed (returns 503) — never accept
    unauthenticated webhooks silently.
    """
    expected = settings.webhook_auth_token
    if not expected:
        if settings.dev_mode:
            logger.warning(
                "Webhook auth bypassed (IDP_DEV_MODE=true) — DEV ONLY",
            )
            return
        # Fail closed: configuration error, not a request error
        logger.error(
            "IDP_WEBHOOK_AUTH_TOKEN is not set in production. "
            "Refusing webhook to prevent unauthenticated access."
        )
        raise HTTPException(
            status_code=503,
            detail="Webhook auth not configured",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1] != expected:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# ---------------------------------------------------------------------------
# Payload schemas
# ---------------------------------------------------------------------------

class PdfAttachedPayload(BaseModel):
    """Payload Salesforce sends when a PDF is attached to a case.

    Salesforce supplies the cert type and funding program directly so the
    IDP doesn't need a follow-up query — those values are already on the
    Case record at the time the webhook fires.
    """
    case_id: str = Field(..., description="Salesforce Case Id (15- or 18-char)")
    case_number: str | None = Field(
        default=None,
        description="Salesforce CaseNumber (e.g. CAS570309) — used in findings text",
    )
    cert_type: str = Field(
        ...,
        description=(
            "Case.CertType__c — one of MI, IC, AR, AR-SC, IR. Drives the "
            "extraction prompts and compliance rules."
        ),
    )
    funding_program: str = Field(
        ...,
        description=(
            "Case.Funding_Program2__c — e.g. LIHTC, HUD, HUD 202/8, USDA. "
            "Drives the work-hours range resolution rule."
        ),
    )
    content_document_id: str | None = Field(
        default=None,
        description=(
            "Optional ContentDocumentId of the attached PDF. If omitted, "
            "the IDP will scan the case for the most recent PDF."
        ),
    )


class MuleSoftDonePayload(BaseModel):
    case_id: str
    case_number: str | None = None


# ---------------------------------------------------------------------------
# Webhook 1: PDF attached
# ---------------------------------------------------------------------------

@router.post(
    "/webhook/pdf-attached",
    status_code=202,
    dependencies=[Depends(_verify_webhook_mode), Depends(_verify_webhook_token)],
)
def pdf_attached(
    payload: PdfAttachedPayload,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Salesforce → IDP: a PDF was attached to a case.

    Salesforce sends case_id + cert_type + funding_program in one payload.
    IDP enqueues the extraction job and returns immediately. PDF download
    happens in the background task.
    """
    logger.info(
        "Webhook pdf-attached: case=%s cert=%s funding=%s document=%s",
        payload.case_number or payload.case_id,
        payload.cert_type,
        payload.funding_program,
        payload.content_document_id,
    )

    store = get_job_store(settings.audit_job_db)
    upsert = store.upsert_pending(
        case_id=payload.case_id,
        case_number=payload.case_number,
        cert_type=payload.cert_type,
        funding_program=payload.funding_program,
        content_document_id=payload.content_document_id,
    )

    if upsert.get("deduplicated"):
        return {
            "status": "already_in_progress",
            "case_id": payload.case_id,
            "state": upsert.get("state"),
        }

    background_tasks.add_task(run_extraction, payload.case_id)

    return {
        "status": "accepted",
        "case_id": payload.case_id,
        "case_number": payload.case_number,
        "cert_type": payload.cert_type,
        "funding_program": payload.funding_program,
    }


# ---------------------------------------------------------------------------
# Webhook 2: MuleSoft done
# ---------------------------------------------------------------------------

@router.post(
    "/webhook/mulesoft-done",
    status_code=202,
    dependencies=[Depends(_verify_webhook_mode), Depends(_verify_webhook_token)],
)
def mulesoft_done(
    payload: MuleSoftDonePayload,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Salesforce → IDP: MuleSoft has finished extracting this case.

    IDP enqueues the comparison job. If the extraction is still running,
    the comparison waits for it to complete.
    """
    logger.info("Webhook mulesoft-done: case_id=%s", payload.case_id)

    store = get_job_store(settings.audit_job_db)
    job = store.get(payload.case_id)

    if job is None:
        # Webhook 2 arrived without webhook 1 — bootstrap a pending job
        # so the comparison can proceed once extraction is triggered.
        # (Comparison will fall through to error if extraction is never
        # triggered — that's correct behavior for an out-of-order signal.)
        logger.warning(
            "mulesoft-done received for unknown case %s — comparing anyway",
            payload.case_id,
        )

    background_tasks.add_task(run_comparison, payload.case_id)

    return {
        "status": "accepted",
        "case_id": payload.case_id,
        "current_state": job.get("state") if job else "unknown",
    }


# ---------------------------------------------------------------------------
# Read endpoint: inspect a case's audit job
# ---------------------------------------------------------------------------

@router.get("/audit/cases/{case_id}")
def get_case_audit(
    case_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return the current state of an audit job, including findings text."""
    store = get_job_store(settings.audit_job_db)
    job = store.get(case_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not in audit store")

    # Don't return the full extraction blob in this endpoint — too large.
    job.pop("extraction_result", None)
    return job
