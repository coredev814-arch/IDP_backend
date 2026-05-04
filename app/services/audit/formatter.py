"""Format ComparisonResult into the IDP_Testing_Results__c text shape.

Two-section layout so analysts can triage at a glance:

    --- AI FILE AUDIT ---
    Case: CAS570309
    Processed: 2026-05-04
    Confidence: 65% (YELLOW)
    Findings: 3 from comparison, 8 from IDP analysis

    === MULESOFT COMPARISON ===
    [CRITICAL]
      - ...
    [REVIEW]
      - ...

    === IDP ANALYSIS ===
    [CRITICAL]
      - ...
    [REVIEW]
      - ...
    [NOTES]
      - ...

Section split is by the *source* of the finding:
  MuleSoft Comparison — produced by diffing AI extraction against MuleSoft
                        records (DUPLICATE_*, MISSING_*, IDP_MISSED_*,
                        VALUE_MISMATCH).
  IDP Analysis        — produced by IDP itself (compliance, signature
                        validation, cross-doc checks, [RED]/[YELLOW]
                        field scoring, extraction notes).

Within each section, findings are bucketed by severity:
  high   -> [CRITICAL]
  medium -> [REVIEW]
  low    -> [NOTES]
  info   -> [NOTES]
"""
from __future__ import annotations

from datetime import date

from app.services.audit.comparator import (
    COMPLIANCE,
    DUPLICATE_ASSET,
    DUPLICATE_INCOME,
    DUPLICATE_MEMBER,
    EXTRACTION_NOTE,
    FIELD_QUALITY,
    IDP_MISSED_ASSET,
    IDP_MISSED_INCOME,
    IDP_MISSED_MEMBER,
    INTERNAL_DISCREPANCY,
    MISSING_ASSET,
    MISSING_INCOME,
    MISSING_MEMBER,
    VALUE_MISMATCH,
    ComparisonResult,
    Finding,
)


# Categories produced by AI ↔ MuleSoft comparison
_MULESOFT_CATEGORIES = frozenset({
    DUPLICATE_MEMBER, DUPLICATE_INCOME, DUPLICATE_ASSET,
    MISSING_MEMBER, MISSING_INCOME, MISSING_ASSET,
    IDP_MISSED_MEMBER, IDP_MISSED_INCOME, IDP_MISSED_ASSET,
    VALUE_MISMATCH,
})

# Categories from IDP's own analysis (no MuleSoft involvement)
_IDP_CATEGORIES = frozenset({
    COMPLIANCE,
    INTERNAL_DISCREPANCY,
    FIELD_QUALITY,
    EXTRACTION_NOTE,
})

_SEVERITY_BUCKETS = {
    "high": "[CRITICAL]",
    "medium": "[REVIEW]",
    "low": "[NOTES]",
    "info": "[NOTES]",
}
_SEVERITY_ORDER = ["high", "medium", "low", "info"]


def format_findings(
    comparison: ComparisonResult,
    case_number: str,
    processed_date: date | None = None,
) -> str:
    """Render the audit summary as plain text for IDP_Testing_Results__c."""
    processed = (processed_date or date.today()).isoformat()
    confidence_pct = int(round(comparison.confidence.case_confidence * 100))
    flag = comparison.confidence.flag.upper()

    mulesoft = [f for f in comparison.findings if f.category in _MULESOFT_CATEGORIES]
    idp = [f for f in comparison.findings if f.category in _IDP_CATEGORIES]
    other = [
        f for f in comparison.findings
        if f.category not in _MULESOFT_CATEGORIES
        and f.category not in _IDP_CATEGORIES
    ]
    # Anything we don't recognize as MuleSoft-side falls under IDP analysis
    idp.extend(other)

    lines: list[str] = [
        "--- AI FILE AUDIT ---",
        f"Case: {case_number}",
        f"Processed: {processed}",
        f"Confidence: {confidence_pct}% ({flag})",
        f"Findings: {len(mulesoft)} from comparison, {len(idp)} from IDP analysis",
        "",
    ]

    if not comparison.findings:
        c = comparison.confidence
        produced_anything = (
            c.agreements > 0 or c.disagreements > 0
            or c.ai_only_records > 0 or c.sf_only_records > 0
        )
        if not produced_anything and c.extraction_score < 0.5:
            lines.append(
                "FINDINGS: extraction did not produce data for comparison "
                "(low extraction quality or unrecognized document). "
                "Manual review required."
            )
        elif not produced_anything:
            lines.append(
                "FINDINGS: no comparable records in either system. "
                "Manual review may be required."
            )
        else:
            lines.append("FINDINGS: none — extraction matches MuleSoft data.")
        return "\n".join(lines)

    if mulesoft:
        lines.append("=== MULESOFT COMPARISON ===")
        lines.extend(_render_section(mulesoft))
        lines.append("")

    if idp:
        lines.append("=== IDP ANALYSIS ===")
        lines.extend(_render_section(idp))
        lines.append("")

    c = comparison.confidence
    lines.append("---")
    lines.append(
        f"Stats: {c.agreements} agreements, "
        f"{c.disagreements} disagreements, "
        f"{c.ai_only_records} AI-only, "
        f"{c.sf_only_records} MuleSoft-only"
    )
    lines.append(
        f"Extraction score: {c.extraction_score:.2f}, "
        f"Agreement rate: {c.agreement_rate:.2f}"
    )
    return "\n".join(lines)


def _render_section(findings: list[Finding]) -> list[str]:
    """Render one section grouped by severity bucket.

    Iterates severities in order (high -> medium -> low -> info), emitting
    the bucket label once even when multiple severities share it
    (low + info both fall under [NOTES]).
    """
    out: list[str] = []
    emitted_buckets: set[str] = set()
    for severity in _SEVERITY_ORDER:
        bucket_label = _SEVERITY_BUCKETS[severity]
        bucket_findings = [f for f in findings if f.severity == severity]
        if not bucket_findings:
            continue
        if bucket_label not in emitted_buckets:
            out.append(bucket_label)
            emitted_buckets.add(bucket_label)
        for f in bucket_findings:
            out.append(f"  - {f.message}")
    return out
