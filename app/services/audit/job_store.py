"""Persistent job state for IDP audit cases.

State machine per case:
  pending           — initial
  extracting        — IDP extraction running
  extracted         — extraction done, awaiting MuleSoft
  comparing         — comparison running
  done              — findings produced
  extraction_failed — IDP extraction errored
  comparison_failed — comparison errored
  mulesoft_timeout  — MuleSoft webhook never arrived

SQLite is sufficient for v1: small write volume, single-process worker,
survives Render restarts when stored in a persistent volume.
"""
import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# State constants
PENDING = "pending"
EXTRACTING = "extracting"
EXTRACTED = "extracted"
COMPARING = "comparing"
DONE = "done"
EXTRACTION_FAILED = "extraction_failed"
COMPARISON_FAILED = "comparison_failed"
MULESOFT_TIMEOUT = "mulesoft_timeout"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_jobs (
    case_id TEXT PRIMARY KEY,
    case_number TEXT,
    state TEXT NOT NULL,
    cert_type TEXT,
    funding_program TEXT,
    content_document_id TEXT,
    extraction_result TEXT,        -- JSON
    findings_text TEXT,
    error TEXT,
    confidence REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    extracted_at REAL,
    mulesoft_done_at REAL,
    completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_state ON audit_jobs(state);
"""


class JobStore:
    """Thread-safe SQLite-backed audit job tracker."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info("Audit job store ready at %s", db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def upsert_pending(
        self,
        case_id: str,
        case_number: str | None,
        cert_type: str | None,
        funding_program: str | None,
        content_document_id: str | None,
    ) -> dict[str, Any]:
        """Create or refresh a pending job. Idempotent.

        If a job already exists in a non-terminal state, return as-is.
        If it's terminal (done/failed), recreate as pending.
        """
        with self._lock, self._connect() as conn:
            now = time.time()
            row = conn.execute(
                "SELECT state FROM audit_jobs WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row and row["state"] in (EXTRACTING, EXTRACTED, COMPARING):
                logger.info(
                    "Job for %s already in state %s — skipping upsert",
                    case_id, row["state"],
                )
                return {"state": row["state"], "deduplicated": True}

            conn.execute("""
                INSERT INTO audit_jobs (
                    case_id, case_number, state, cert_type, funding_program,
                    content_document_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    case_number = excluded.case_number,
                    state = excluded.state,
                    cert_type = excluded.cert_type,
                    funding_program = excluded.funding_program,
                    content_document_id = excluded.content_document_id,
                    extraction_result = NULL,
                    findings_text = NULL,
                    error = NULL,
                    confidence = NULL,
                    updated_at = excluded.updated_at,
                    extracted_at = NULL,
                    mulesoft_done_at = NULL,
                    completed_at = NULL
            """, (
                case_id, case_number, PENDING, cert_type, funding_program,
                content_document_id, now, now,
            ))
            return {"state": PENDING, "deduplicated": False}

    def mark_extracting(self, case_id: str) -> None:
        self._set_state(case_id, EXTRACTING)

    def mark_extracted(self, case_id: str, extraction_result: dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            now = time.time()
            conn.execute("""
                UPDATE audit_jobs
                SET state = ?, extraction_result = ?,
                    extracted_at = ?, updated_at = ?
                WHERE case_id = ?
            """, (
                EXTRACTED, json.dumps(extraction_result, default=str),
                now, now, case_id,
            ))

    def mark_extraction_failed(self, case_id: str, error: str) -> None:
        self._set_state(case_id, EXTRACTION_FAILED, error=error)

    def mark_mulesoft_done(self, case_id: str) -> None:
        """Record that MuleSoft signaled completion. INSERTs a stub if the
        case is unknown so the signal isn't lost when webhook 2 races
        ahead of webhook 1.
        """
        with self._lock, self._connect() as conn:
            now = time.time()
            cur = conn.execute(
                "UPDATE audit_jobs SET mulesoft_done_at = ?, updated_at = ? "
                "WHERE case_id = ?",
                (now, now, case_id),
            )
            if cur.rowcount == 0:
                # Out-of-order signal: webhook 2 arrived before webhook 1.
                # Insert a stub so the timestamp is preserved and a later
                # extraction trigger can find it.
                conn.execute("""
                    INSERT INTO audit_jobs (
                        case_id, state, created_at, updated_at, mulesoft_done_at
                    ) VALUES (?, ?, ?, ?, ?)
                """, (case_id, PENDING, now, now, now))
                logger.warning(
                    "mark_mulesoft_done: case %s unknown — created stub "
                    "to preserve signal", case_id,
                )

    def try_claim_for_comparison(self, case_id: str) -> bool:
        """Atomically transition EXTRACTED → COMPARING. Returns True if this
        caller won the claim. Used to prevent duplicate concurrent comparisons
        when webhook 2 and the post-extraction trigger race.
        """
        with self._lock, self._connect() as conn:
            now = time.time()
            cur = conn.execute("""
                UPDATE audit_jobs SET state = ?, updated_at = ?
                WHERE case_id = ? AND state = ?
            """, (COMPARING, now, case_id, EXTRACTED))
            return cur.rowcount > 0

    def mark_comparing(self, case_id: str) -> None:
        self._set_state(case_id, COMPARING)

    def mark_done(
        self,
        case_id: str,
        findings_text: str,
        confidence: float,
    ) -> None:
        with self._lock, self._connect() as conn:
            now = time.time()
            conn.execute("""
                UPDATE audit_jobs
                SET state = ?, findings_text = ?, confidence = ?,
                    completed_at = ?, updated_at = ?
                WHERE case_id = ?
            """, (DONE, findings_text, confidence, now, now, case_id))

    def mark_comparison_failed(self, case_id: str, error: str) -> None:
        self._set_state(case_id, COMPARISON_FAILED, error=error)

    def _set_state(self, case_id: str, state: str, error: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            now = time.time()
            conn.execute("""
                UPDATE audit_jobs SET state = ?, updated_at = ?, error = ?
                WHERE case_id = ?
            """, (state, now, error, case_id))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audit_jobs WHERE case_id = ?", (case_id,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            if data.get("extraction_result"):
                try:
                    data["extraction_result"] = json.loads(data["extraction_result"])
                except json.JSONDecodeError:
                    pass
            return data

    def list_by_state(self, state: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_jobs WHERE state = ? ORDER BY updated_at",
                (state,),
            ).fetchall()
            return [dict(r) for r in rows]


_SINGLETON: JobStore | None = None
_SINGLETON_LOCK = threading.Lock()


def get_job_store(db_path: Path) -> JobStore:
    """Module-level singleton so all webhook handlers share one store."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = JobStore(db_path)
    return _SINGLETON
