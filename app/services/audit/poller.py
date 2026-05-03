"""Polling-mode trigger for the IDP audit pipeline.

Alternative to webhooks: instead of Salesforce pushing signals to IDP,
IDP queries Salesforce on a fixed interval for cases that are ready
(MuleSoft processing complete) and runs the audit on each.

In polling mode there's no separate "webhook 2" — by the time the poller
finds a case, MuleSoft is already done. So extraction and comparison run
back-to-back as part of one job.

Two ways to run:
  1. Standalone — `python -m scripts.run_audit_poller` (separate process)
  2. Embedded — auto-started inside FastAPI lifespan when audit_mode is
     "poll" or "both". See `start_poller_thread` and `stop_poller_thread`.
"""
from __future__ import annotations

import logging
import threading
import time

from app.core.config import Settings
from app.core.dependencies import get_settings
from app.services.audit.job_store import (
    COMPARING,
    DONE,
    EXTRACTED,
    EXTRACTING,
    get_job_store,
)
from app.services.audit.jobs import run_comparison, run_extraction
from app.services.salesforce.client import _escape_soql, get_salesforce_client

logger = logging.getLogger(__name__)


# SOQL `IN` clauses are limited (default ~4 KB query length, practical
# safe ceiling ~200 IDs). We chunk the exclusion list if it grows beyond.
_MAX_EXCLUDE_IDS_PER_QUERY = 200


def _build_ready_query(limit: int, exclude_ids: set[str]) -> str:
    """Build the SOQL query, excluding already-known case IDs.

    Cases stay marked `IDP_File_Process_Status__c = 'Processed'` forever
    in Salesforce, so without this filter the poller would re-fetch the
    same cases on every cycle. The exclusion list comes from the local
    JobStore so we never re-process cases already in flight or completed.

    Audit window — only process cases between MuleSoft completion and
    analyst review:
      - IDP_File_Process_Status__c = 'Processed' (MuleSoft done)
      - IDP_Phase_1_Review__c = true             (Phase 1 ready)
      - IDP_Process_Complete__c != null          (timestamp set)
      - IDP_Meez_Review_Completed__c = null      (analyst hasn't reviewed)
      - Final_Approval__c = null                 (case not finalized)
    Without the last two filters, IDP would re-audit cases that humans
    have already reviewed and closed — producing useless findings on
    summary/cover-sheet PDFs uploaded after the original packet.
    """
    where_parts = [
        "IDP_File_Process_Status__c = 'Processed'",
        "IDP_Phase_1_Review__c = true",
        "IDP_Process_Complete__c != null",
        "IDP_Meez_Review_Completed__c = null",
        "Final_Approval__c = null",
    ]
    if exclude_ids:
        # Limit to a safe number of IDs per query — extra cases will be
        # caught on subsequent poll cycles as the exclusion list shrinks.
        ids_subset = list(exclude_ids)[:_MAX_EXCLUDE_IDS_PER_QUERY]
        ids_str = ", ".join(f"'{_escape_soql(i)}'" for i in ids_subset)
        where_parts.append(f"Id NOT IN ({ids_str})")

    where_clause = "\n  AND ".join(where_parts)
    return f"""
SELECT Id, CaseNumber, CertType__c, Funding_Program2__c
FROM Case
WHERE {where_clause}
ORDER BY IDP_Process_Complete__c ASC
LIMIT {limit}
""".strip()


def fetch_ready_cases(
    settings: Settings,
    limit: int,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Query Salesforce for ready cases, excluding already-known IDs."""
    sf = get_salesforce_client(settings)
    query = _build_ready_query(limit, exclude_ids or set())
    result = sf.sf.query(query)
    return result.get("records", [])


def is_already_processed(case_id: str, settings: Settings) -> bool:
    """Skip cases the local JobStore already has in a non-restartable state."""
    store = get_job_store(settings.audit_job_db)
    job = store.get(case_id)
    if not job:
        return False
    # Reprocess only if the previous run failed
    return job["state"] in (EXTRACTING, EXTRACTED, COMPARING, DONE)


def process_case(case_record: dict, settings: Settings) -> None:
    """Run the full audit for one case: extraction → comparison.

    Sequential by design — in poll mode there's no separate MuleSoft signal,
    so we trigger comparison immediately after extraction completes.
    """
    case_id = case_record["Id"]
    case_number = case_record.get("CaseNumber")
    cert_type = case_record.get("CertType__c")
    funding = case_record.get("Funding_Program2__c")

    # Cases without a CertType__c are usually stubs or non-cert work items
    # that shouldn't be audited. Mark as failed so we don't keep retrying.
    if not cert_type:
        logger.warning(
            "Skipping case=%s — CertType__c is missing on the Case record. "
            "Polling SOQL should typically exclude these; flagging as failed.",
            case_number or case_id,
        )
        store = get_job_store(settings.audit_job_db)
        store.upsert_pending(
            case_id=case_id,
            case_number=case_number,
            cert_type=None,
            funding_program=None,
            content_document_id=None,
        )
        store.mark_extraction_failed(
            case_id, "CertType__c missing — case not eligible for audit",
        )
        return

    logger.info(
        "Polling: processing case=%s cert=%s funding=%s",
        case_number, cert_type, funding,
    )

    store = get_job_store(settings.audit_job_db)
    upsert = store.upsert_pending(
        case_id=case_id,
        case_number=case_number,
        cert_type=cert_type,
        funding_program=funding,
        content_document_id=None,    # poller resolves PDF via case scan
    )
    if upsert.get("deduplicated"):
        logger.info("Case %s already in progress — skipping", case_number)
        return

    # In poll mode MuleSoft is already done — record the signal so the
    # post-extraction comparison fires automatically.
    store.mark_mulesoft_done(case_id)

    try:
        run_extraction(case_id)
    except Exception:
        logger.exception("Extraction failed for case %s in poll mode", case_number)
        return

    # Extraction triggers comparison automatically when mulesoft_done_at
    # was already set (see run_extraction). If for some reason it didn't,
    # call comparison explicitly as a safety net.
    refreshed = store.get(case_id)
    if refreshed and refreshed["state"] == EXTRACTED:
        run_comparison(case_id)


def poll_once(settings: Settings) -> int:
    """Run one polling cycle. Returns count of cases processed.

    Builds the exclusion list from the local JobStore — already-known
    case IDs (in any state except failed) are filtered out at the SOQL
    level so we don't waste API calls on cases we've already handled.

    Failed cases (extraction/comparison/timeout) are NOT excluded — they
    get retried on the next poll cycle.
    """
    store = get_job_store(settings.audit_job_db)
    known = store.list_known_case_ids()
    failed = store.list_failed_case_ids()
    exclude_ids = known - failed   # exclude all known except failed (retry those)

    cases = fetch_ready_cases(
        settings,
        settings.audit_poll_batch_size,
        exclude_ids=exclude_ids,
    )
    if not cases:
        return 0

    logger.info(
        "Poll cycle: fetched %d ready case(s) (excluded %d known IDs, %d failed eligible for retry)",
        len(cases), len(exclude_ids), len(failed),
    )

    processed = 0
    for case in cases:
        # Defense-in-depth: re-check locally after Salesforce returns
        # results, in case another worker claimed it between query and
        # processing (multi-worker scenarios).
        if is_already_processed(case["Id"], settings):
            logger.debug(
                "Skipping case %s — already in non-restartable state",
                case.get("CaseNumber") or case["Id"],
            )
            continue
        try:
            process_case(case, settings)
            processed += 1
        except Exception:
            logger.exception(
                "Unhandled error processing case %s",
                case.get("CaseNumber") or case.get("Id"),
            )
    return processed


def run_poll_loop(stop_event: threading.Event | None = None) -> None:
    """Main entry point — runs until stop_event is set (or forever).

    When called from a standalone script, leave stop_event as None — the
    loop runs until process termination. When called from FastAPI lifespan,
    pass a threading.Event so shutdown can request a clean exit.
    """
    settings = get_settings()
    interval = settings.audit_poll_interval_seconds
    logger.info(
        "IDP audit poller started (mode=%s, interval=%ds, batch=%d)",
        settings.audit_mode, interval, settings.audit_poll_batch_size,
    )

    while True:
        if stop_event is not None and stop_event.is_set():
            logger.info("Poller stop requested — exiting loop")
            return
        try:
            n = poll_once(settings)
            if n > 0:
                logger.info("Poll cycle processed %d case(s)", n)
        except Exception:
            logger.exception("Poll cycle failed")

        # Sleep in short ticks so a stop request is respected promptly
        # rather than waiting up to a full interval.
        if stop_event is not None:
            if stop_event.wait(interval):
                logger.info("Poller stop requested during sleep — exiting")
                return
        else:
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Embedded mode — start/stop the poller as a background thread inside
# the FastAPI process so a single Render service can serve both webhooks
# AND poll Salesforce on a schedule.
# ---------------------------------------------------------------------------

_THREAD: threading.Thread | None = None
_STOP_EVENT: threading.Event | None = None


def start_poller_thread() -> bool:
    """Start the poll loop on a daemon thread. Idempotent.

    Returns True if a new thread was started, False if one was already
    running or the configured mode does not enable polling.
    """
    global _THREAD, _STOP_EVENT

    settings = get_settings()
    if settings.audit_mode not in ("poll", "both"):
        logger.info(
            "Poller not started (audit_mode=%s) — set IDP_AUDIT_MODE=poll "
            "or =both to enable in-process polling",
            settings.audit_mode,
        )
        return False

    if _THREAD is not None and _THREAD.is_alive():
        logger.warning("start_poller_thread called but poller already running")
        return False

    _STOP_EVENT = threading.Event()
    _THREAD = threading.Thread(
        target=run_poll_loop,
        args=(_STOP_EVENT,),
        name="idp-audit-poller",
        daemon=True,
    )
    _THREAD.start()
    logger.info("Audit poller thread started (daemon)")
    return True


def stop_poller_thread(timeout: float = 5.0) -> None:
    """Signal the poll loop to stop and wait briefly for it to exit."""
    global _THREAD, _STOP_EVENT
    if _STOP_EVENT is None or _THREAD is None:
        return
    _STOP_EVENT.set()
    _THREAD.join(timeout=timeout)
    if _THREAD.is_alive():
        logger.warning(
            "Poller thread did not exit within %.1fs — leaving as daemon",
            timeout,
        )
    else:
        logger.info("Audit poller thread stopped cleanly")
    _THREAD = None
    _STOP_EVENT = None
