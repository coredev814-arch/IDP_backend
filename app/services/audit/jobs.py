"""Background jobs for the IDP audit pipeline.

Two jobs:
  run_extraction(case_id) — webhook 1 trigger; downloads PDF + runs IDP.
  run_comparison(case_id) — webhook 2 trigger; pulls MuleSoft data + diffs.

Both are designed to be idempotent and safe to retry. State is persisted
in the JobStore so repeated webhooks don't double-run work.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import date

from app.core.config import Settings
from app.core.dependencies import get_settings
from app.services.audit.comparator import compare
from app.services.audit.formatter import format_findings
from app.services.audit.job_store import (
    COMPARING,
    DONE,
    EXTRACTED,
    EXTRACTING,
    EXTRACTION_FAILED,
    JobStore,
    get_job_store,
)
from app.services.pdf_service import process_pdf_full
from app.services.salesforce.client import get_salesforce_client

logger = logging.getLogger(__name__)


def _store(settings: Settings) -> JobStore:
    return get_job_store(settings.audit_job_db)


# ---------------------------------------------------------------------------
# Extraction job (triggered by webhook 1: PDF attached)
# ---------------------------------------------------------------------------

def run_extraction(case_id: str) -> None:
    """Download PDF, run IDP pipeline, store extraction result.

    Idempotent: if the job is already in extracting/extracted/comparing/done,
    this is a no-op. Run via FastAPI BackgroundTasks.
    """
    settings = get_settings()
    store = _store(settings)

    job = store.get(case_id)
    if job is None:
        logger.warning("run_extraction: no job entry for %s — aborting", case_id)
        return
    if job["state"] in (EXTRACTING, EXTRACTED, COMPARING, DONE):
        logger.info(
            "run_extraction: %s already in state %s — skipping",
            case_id, job["state"],
        )
        return

    sf = get_salesforce_client(settings)
    store.mark_extracting(case_id)

    try:
        # Resolve PDF: prefer the content_document_id from the webhook,
        # fall back to scanning the case for the most recent attachment.
        cd_id = job.get("content_document_id")
        if cd_id:
            pdf_bytes = sf.download_pdf(cd_id)
        else:
            pdf_bytes, cd_id = sf.get_pdf_for_case(case_id)
            logger.info("Resolved PDF for %s via case scan: %s", case_id, cd_id)

        cert_type = job.get("cert_type") or None
        funding = job.get("funding_program") or None

        logger.info(
            "Running IDP extraction for case=%s cert=%s funding=%s",
            job.get("case_number"), cert_type, funding,
        )
        result = process_pdf_full(
            pdf_bytes,
            settings,
            funding_program=funding,
            certification_type=cert_type,
        )

        # Convert pydantic ExtractionResult to dict for storage
        extraction = result["extraction"]
        if hasattr(extraction, "model_dump"):
            extraction_dict = extraction.model_dump()
        else:
            extraction_dict = extraction

        store.mark_extracted(case_id, extraction_dict)
        logger.info(
            "Extraction complete for %s — %d pages, overall_flag=%s",
            case_id,
            len((extraction_dict.get("classification") or {}).get("pages") or []),
            (extraction_dict.get("field_scores") or {}).get("overall_flag"),
        )

        # If MuleSoft already finished (webhook 2 arrived during extraction),
        # the job will have mulesoft_done_at set — kick off comparison now.
        refreshed = store.get(case_id)
        if refreshed and refreshed.get("mulesoft_done_at"):
            run_comparison(case_id)

    except Exception as exc:
        logger.exception("Extraction failed for case %s", case_id)
        store.mark_extraction_failed(case_id, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Comparison job (triggered by webhook 2: MuleSoft done)
# ---------------------------------------------------------------------------

def run_comparison(case_id: str) -> None:
    """Pull MuleSoft data, compare against stored IDP extraction, build findings.

    If the IDP extraction isn't done yet, polls until it completes (or
    times out after audit_extraction_wait_seconds).

    Note: writing findings back to Salesforce is intentionally NOT done here
    yet — the formatted text is logged and stored in JobStore. Once the
    Salesforce writeback is approved, add the sf.update() call at the bottom.
    """
    settings = get_settings()
    store = _store(settings)

    # Mark MuleSoft signal received (does NOT change state — just records timing)
    store.mark_mulesoft_done(case_id)

    job = store.get(case_id)
    if job is None:
        logger.warning("run_comparison: no job entry for %s — aborting", case_id)
        return
    if job["state"] in (COMPARING, DONE):
        logger.info(
            "run_comparison: %s already in state %s — skipping",
            case_id, job["state"],
        )
        return

    # Wait for extraction to complete
    if job["state"] != EXTRACTED:
        deadline = time.time() + settings.audit_extraction_wait_seconds
        logger.info(
            "run_comparison: %s not yet extracted (state=%s) — waiting",
            case_id, job["state"],
        )
        while time.time() < deadline:
            time.sleep(settings.audit_extraction_poll_seconds)
            job = store.get(case_id)
            if job is None:
                return
            if job["state"] == EXTRACTED:
                break
            if job["state"] == EXTRACTION_FAILED:
                logger.error(
                    "run_comparison: extraction failed for %s — cannot compare",
                    case_id,
                )
                return
        else:
            logger.warning(
                "run_comparison: extraction wait timeout for %s (state=%s)",
                case_id, job["state"],
            )
            return

    # Atomic claim — prevents duplicate comparisons when webhook 2 and
    # the post-extraction trigger race. Only one caller wins.
    if not store.try_claim_for_comparison(case_id):
        logger.info(
            "run_comparison: another worker claimed comparison for %s — "
            "this caller bails out", case_id,
        )
        return

    try:
        sf = get_salesforce_client(settings)
        sf_data = sf.get_case_audit_data(case_id)

        extraction = job["extraction_result"]
        if not isinstance(extraction, dict):
            raise ValueError("Stored extraction is not a dict")

        comparison = compare(extraction, sf_data)
        case_number = job.get("case_number") or case_id

        findings_text = format_findings(
            comparison, case_number=case_number, processed_date=date.today(),
        )

        store.mark_done(
            case_id,
            findings_text=findings_text,
            confidence=comparison.confidence.case_confidence,
        )

        logger.info(
            "Audit complete for case=%s confidence=%.2f flag=%s findings=%d",
            case_number,
            comparison.confidence.case_confidence,
            comparison.confidence.flag,
            len(comparison.findings),
        )

        # Salesforce writeback intentionally deferred. The findings text is
        # available via the JobStore (state=done) and can be inspected via
        # the GET /audit/cases/{case_id} endpoint or directly in SQLite.
        logger.info(
            "Findings (not yet pushed to Salesforce):\n%s\n%s\n%s",
            "=" * 60, findings_text, "=" * 60,
        )

    except Exception as exc:
        logger.exception("Comparison failed for case %s", case_id)
        store.mark_comparison_failed(case_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
