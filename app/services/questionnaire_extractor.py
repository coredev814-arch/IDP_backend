"""Questionnaire disclosure extractor — LLM extraction of yes/no disclosures (Section 11)."""

import logging

from app.core.config import Settings
from app.schemas.extraction import DocumentGroup, QuestionnaireDisclosures
from app.services.llm_service import call_llm_json

logger = logging.getLogger(__name__)

QUESTIONNAIRE_DISCLOSURE_PROMPT = """\
You are an expert data extractor for HUD/Affordable Housing application forms.

Extract YES/NO disclosures from the provided application, questionnaire, or \
self-certification form. These are questions the applicant answered about their \
income sources, assets, and status.

FIELDS TO EXTRACT (all boolean — true if disclosed/affirmed, false if denied, null if not asked):
- has_employment: Does the applicant report having a job or employment income?
- employers: List of employer names mentioned (empty list if none). Title Case.
- has_student_status: Does any household member report being a student?
- has_ssa_benefits: Does the applicant report receiving Social Security (SSA/SSI/SSDI)?
- has_checking_account: Does the applicant report having a checking account?
- has_savings_account: Does the applicant report having a savings account?
- has_child_support: Does the applicant report receiving or paying child support/alimony?
- has_pension: Does the applicant report receiving pension or retirement income?
- has_self_employment: Does the applicant report self-employment income?
- has_other_income: Does the applicant report other income (tips, gifts, TANF, etc.)?
- has_real_estate: Does the applicant report owning real estate?
- has_life_insurance: Does the applicant report having life insurance (whole life)?

RULES:
- Look for checkboxes, yes/no answers, circled responses, listed amounts
- If a dollar amount > 0 is listed next to a source, that counts as "true"
- If a field is left blank or the question wasn't asked, set to null
- "Do you have a checking account? Yes, 1 account, $500" → has_checking_account = true
- "Employment: None" or "N/A" → has_employment = false
- Extract employer names exactly as written, in Title Case

Return ONLY valid JSON matching the schema above."""


def extract_questionnaire_disclosures(
    groups: list[DocumentGroup],
    settings: Settings,
) -> QuestionnaireDisclosures | None:
    """Extract yes/no disclosures from application/questionnaire documents.

    Returns QuestionnaireDisclosures or None if no questionnaire found.
    """
    relevant_texts = []
    for g in groups:
        if g.category == "ignore":
            continue
        dt = g.document_type.lower()
        if "application" in dt or "questionnaire" in dt or "self-certification" in dt:
            relevant_texts.append(
                f"[Document: {g.document_type}, Pages: {g.page_range}]\n{g.combined_text}"
            )

    if not relevant_texts:
        logger.info("No questionnaire documents found for disclosure extraction")
        return None

    user_prompt = (
        "Extract yes/no disclosures from these application/questionnaire documents:\n\n"
        + "\n\n---\n\n".join(relevant_texts)
    )

    result = call_llm_json(QUESTIONNAIRE_DISCLOSURE_PROMPT, user_prompt, settings)
    logger.info("Extracted questionnaire disclosures")
    return QuestionnaireDisclosures.model_validate(result)


def validate_affirmative_responses(
    disclosures: QuestionnaireDisclosures | None,
    document_groups: list[DocumentGroup],
    income_doc_types: set[str] | None = None,
    asset_doc_types: set[str] | None = None,
) -> list[str]:
    """Cross-reference disclosures against documents present (Affirmative Response Rule).

    Per Section 11: any affirmative response requires independent verification.

    Returns list of compliance finding strings.
    """
    if not disclosures:
        return []

    findings: list[str] = []

    # Build sets of document types present (non-ignore)
    doc_types = {g.document_type for g in document_groups if g.category != "ignore"}
    doc_types_lower = {dt.lower() for dt in doc_types}

    # Employment → VOI + paystubs required
    if disclosures.has_employment is True:
        has_voi = any("voi" in dt or "verification of income" in dt for dt in doc_types_lower)
        has_paystub = any("paystub" in dt or "pay stub" in dt or "pay-slip" in dt for dt in doc_types_lower)
        has_work_number = any("work number" in dt or "equifax" in dt for dt in doc_types_lower)
        if not has_voi and not has_work_number:
            findings.append(
                "Employment disclosed on questionnaire but no Verification of Income (VOI) "
                "or Work Number report found — independent verification required (Section 11)"
            )
        if not has_paystub and not has_work_number:
            findings.append(
                "Employment disclosed on questionnaire but no pay stubs or Work Number report "
                "found — pay stubs required for employment verification (Section 11)"
            )

    # Student status → Student Status Certification required
    if disclosures.has_student_status is True:
        has_student_cert = any("student" in dt for dt in doc_types_lower)
        if not has_student_cert:
            findings.append(
                "Student status disclosed on questionnaire but no Student Status Certification "
                "found — verification required (Section 11)"
            )

    # SSA benefits → SSA Benefit Letter required
    if disclosures.has_ssa_benefits is True:
        has_ssa = any("ssa" in dt or "ssi" in dt or "ssdi" in dt or "social security" in dt
                      for dt in doc_types_lower)
        if not has_ssa:
            findings.append(
                "SSA/SSI/SSDI benefits disclosed on questionnaire but no benefit letter "
                "found — independent verification required (Section 11)"
            )

    # Checking account → bank verification required
    if disclosures.has_checking_account is True:
        has_bank = any("bank statement" in dt or "voa" in dt or "verification of asset" in dt
                       for dt in doc_types_lower)
        if not has_bank:
            findings.append(
                "Checking account disclosed on questionnaire but no bank statement or VOA "
                "found — bank verification required (Section 11)"
            )

    # Savings account → bank verification required
    if disclosures.has_savings_account is True:
        has_bank = any("bank statement" in dt or "voa" in dt or "verification of asset" in dt
                       for dt in doc_types_lower)
        if not has_bank:
            findings.append(
                "Savings account disclosed on questionnaire but no bank statement or VOA "
                "found — bank verification required (Section 11)"
            )

    # Child support → verification required
    if disclosures.has_child_support is True:
        has_cs = any("child support" in dt for dt in doc_types_lower)
        if not has_cs:
            findings.append(
                "Child support disclosed on questionnaire but no child support verification "
                "found — independent verification required (Section 11)"
            )

    # Pension → pension statement required
    if disclosures.has_pension is True:
        has_pension = any("pension" in dt for dt in doc_types_lower)
        if not has_pension:
            findings.append(
                "Pension income disclosed on questionnaire but no pension statement "
                "found — independent verification required (Section 11)"
            )

    # Real estate → documentation required
    if disclosures.has_real_estate is True:
        has_real_estate = any("real estate" in dt for dt in doc_types_lower)
        if not has_real_estate:
            findings.append(
                "Real estate ownership disclosed on questionnaire but no real estate "
                "documentation found — verification required (Section 11)"
            )

    # Life insurance → verification required
    if disclosures.has_life_insurance is True:
        has_life = any("life insurance" in dt for dt in doc_types_lower)
        if not has_life:
            findings.append(
                "Life insurance disclosed on questionnaire but no life insurance "
                "documentation found — verification required (Section 11)"
            )

    return findings
