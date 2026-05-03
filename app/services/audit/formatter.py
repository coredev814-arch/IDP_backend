"""Format ComparisonResult into the IDP_Testing_Results__c text shape.

Produces the layout the analysts expect:

    --- AI FILE AUDIT ---
    Case: CAS570309
    Processed: 2026-05-03
    Confidence: 82%

    FINDINGS:

    1. DUPLICATE MEMBER — ...
    2. INCOME MISMATCH — ...
    ...
"""
from __future__ import annotations

from datetime import date

from app.services.audit.comparator import ComparisonResult, Finding


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def format_findings(
    comparison: ComparisonResult,
    case_number: str,
    processed_date: date | None = None,
) -> str:
    """Render the audit summary as plain text."""
    processed = (processed_date or date.today()).isoformat()
    confidence_pct = int(round(comparison.confidence.case_confidence * 100))

    lines: list[str] = [
        "--- AI FILE AUDIT ---",
        f"Case: {case_number}",
        f"Processed: {processed}",
        f"Confidence: {confidence_pct}%",
        f"Flag: {comparison.confidence.flag.upper()}",
        "",
    ]

    if not comparison.findings:
        lines.append("FINDINGS: none — extraction matches MuleSoft data.")
        return "\n".join(lines)

    sorted_findings = sorted(
        comparison.findings,
        key=lambda f: _SEVERITY_ORDER.get(f.severity, 99),
    )

    lines.append("FINDINGS:")
    lines.append("")
    for i, f in enumerate(sorted_findings, 1):
        lines.extend(_render_finding(i, f))
        lines.append("")

    lines.append("---")
    c = comparison.confidence
    lines.append(
        f"Comparison stats: {c.agreements} agreements · "
        f"{c.disagreements} disagreements · "
        f"{c.ai_only_records} AI-only · "
        f"{c.sf_only_records} MuleSoft-only"
    )
    lines.append(
        f"Extraction score: {c.extraction_score:.2f} · "
        f"Agreement rate: {c.agreement_rate:.2f}"
    )
    return "\n".join(lines)


def _render_finding(idx: int, finding: Finding) -> list[str]:
    """One finding rendered as 1-3 indented lines."""
    return [f"{idx}. {finding.message}"]
