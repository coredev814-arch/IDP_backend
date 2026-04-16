"""Certification type-specific processing rules (Section 12)."""

import logging

from app.schemas.extraction import (
    DocumentGroup,
    DocumentInventory,
    HouseholdDemographics,
)

logger = logging.getLogger(__name__)


def validate_cert_type_requirements(
    cert_type: str | None,
    document_groups: list[DocumentGroup],
    inventory_hud: DocumentInventory | None,
    household: HouseholdDemographics | None,
) -> list[str]:
    """Check certification-type-specific document requirements per Section 12.

    Returns list of compliance finding strings.
    """
    if not cert_type:
        return []

    findings: list[str] = []
    doc_types = {g.document_type for g in document_groups if g.category != "ignore"}
    member_count = len(household.houseHold) if household and household.houseHold else 0

    hud_doc_types = set()
    if inventory_hud:
        for doc in inventory_hud.documents:
            hud_doc_types.add(doc.documentType or "")

    ct = cert_type.upper()

    if ct in ("MI", "IC"):
        findings.extend(_check_mi_requirements(doc_types, hud_doc_types, member_count))
    elif ct == "AR":
        findings.extend(_check_ar_requirements(doc_types, document_groups))
    elif ct == "AR-SC":
        findings.extend(_check_arsc_requirements(doc_types))
    elif ct == "IR":
        findings.extend(_check_ir_requirements())

    return findings


def _check_mi_requirements(
    doc_types: set[str],
    hud_doc_types: set[str],
    member_count: int,
) -> list[str]:
    """Move-In / Initial Certification requires additional documents."""
    findings: list[str] = []

    # Citizenship Declaration (Section 214) — one per member
    has_citizenship = any("Citizenship" in dt or "Section 214" in dt for dt in doc_types | hud_doc_types)
    if not has_citizenship:
        findings.append(
            "MI/IC certification requires Citizenship Declaration (Section 214) "
            "for each household member — not found (Section 12)"
        )

    # Race and Ethnic Data Form — one per member
    has_race = any("Race" in dt and "Ethnic" in dt for dt in doc_types | hud_doc_types)
    if not has_race:
        findings.append(
            "MI/IC certification requires Race and Ethnic Data Form "
            "for each household member — not found (Section 12)"
        )

    # Owner Summary Sheet
    has_owner_summary = any("Owner Summary" in dt for dt in doc_types)
    if not has_owner_summary:
        findings.append(
            "MI/IC certification requires Owner Summary Sheet — not found (Section 12)"
        )

    # Family Summary Sheet
    has_family_summary = any("Family Summary" in dt for dt in doc_types)
    if not has_family_summary:
        findings.append(
            "MI/IC certification requires Family Summary Sheet — not found (Section 12)"
        )

    # Application for Housing
    has_application = any("Application" in dt or "Questionnaire" in dt for dt in doc_types)
    if not has_application:
        findings.append(
            "MI/IC certification requires Application for Housing — not found (Section 12)"
        )

    # HUD 92006 (Emergency Contact)
    has_92006 = any("92006" in dt for dt in doc_types | hud_doc_types)
    if not has_92006:
        findings.append(
            "MI/IC certification requires HUD 92006 (Emergency Contact) — not found (Section 12)"
        )

    return findings


def _check_ar_requirements(
    doc_types: set[str],
    document_groups: list[DocumentGroup],
) -> list[str]:
    """Annual Recertification — previous cert must exist for comparison."""
    findings: list[str] = []

    # Previous certification should exist
    has_previous = any("(Previous)" in g.document_type for g in document_groups)
    if not has_previous:
        findings.append(
            "AR certification but no previous certification found for comparison — "
            "previous cert is expected for annual recertification (Section 12)"
        )

    return findings


def _check_arsc_requirements(doc_types: set[str]) -> list[str]:
    """AR-SC — TIC must exist (it's the source of truth)."""
    findings: list[str] = []

    has_tic = any("TIC" in dt or "Tenant Income Certification" in dt for dt in doc_types)
    if not has_tic:
        findings.append(
            "AR-SC certification requires TIC form (source of truth for self-certification) — "
            "not found (Section 12/13)"
        )

    return findings


def _check_ir_requirements() -> list[str]:
    """Interim Recertification — informational finding."""
    return [
        "IR (Interim Recertification) detected — review is scoped to the specific change "
        "that triggered the interim (e.g., new employment, loss of income, new household member) (Section 12)"
    ]
