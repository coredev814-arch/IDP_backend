"""Questionnaire/application disclosure parser — keyword-based extraction."""

import logging
import re

from app.schemas.extraction import DocumentGroup, QuestionnaireDisclosures
from app.services.parsers.source_normalizer import normalize_source_name

logger = logging.getLogger(__name__)


def parse_questionnaire(groups: list[DocumentGroup]) -> QuestionnaireDisclosures | None:
    """Extract disclosures from application/questionnaire using keyword matching.

    Handles both full applications and Interim Recertification Reports.

    Returns QuestionnaireDisclosures or None if no relevant documents.
    """
    relevant_texts = []
    for g in groups:
        if g.category == "ignore":
            continue
        dt = g.document_type.lower()
        if "application" in dt or "questionnaire" in dt or "recertification report" in dt:
            relevant_texts.append(g.combined_text)

    if not relevant_texts:
        return None

    combined = "\n".join(relevant_texts)
    combined_lower = combined.lower()

    disclosures = QuestionnaireDisclosures()

    # --- Employment ---
    disclosures.has_employment, disclosures.employers = _check_employment(combined, combined_lower)

    # --- Student status ---
    disclosures.has_student_status = _check_keyword(
        combined_lower,
        positive=["student", "enrolled", "attending school", "full-time student", "part-time student"],
        negative=["not a student", "no student"],
    )

    # --- SSA benefits ---
    disclosures.has_ssa_benefits = _check_keyword(
        combined_lower,
        positive=[
            "receive social security", "social security benefit", "social security income",
            "social security amount", "ssa benefit", "ssi benefit", "ssdi benefit",
            "disability benefit", "monthly social security",
        ],
        negative=[
            "no social security", "does not receive",
            "misusing the social security number",  # consent form boilerplate
            "social security number", "social security act",  # legal references, not income
            "penalty provisions",
        ],
    )

    # --- Checking account ---
    disclosures.has_checking_account = _check_keyword(
        combined_lower,
        positive=["checking account"],
        negative=["no checking", "do not have a checking"],
    )

    # --- Savings account ---
    disclosures.has_savings_account = _check_keyword(
        combined_lower,
        positive=["savings account"],
        negative=["no savings", "do not have a savings"],
    )

    # --- Child support ---
    disclosures.has_child_support = _check_keyword(
        combined_lower,
        positive=["child support", "alimony"],
        negative=["no child support", "does not receive child support"],
    )

    # --- Pension ---
    disclosures.has_pension = _check_keyword(
        combined_lower,
        positive=["pension", "retirement income", "retirement benefit"],
        negative=["no pension"],
    )

    # --- Self-employment ---
    disclosures.has_self_employment = _check_keyword(
        combined_lower,
        positive=["self-employed", "self employed", "own business", "freelance"],
        negative=["not self-employed"],
    )

    # --- Other income ---
    disclosures.has_other_income = _check_keyword(
        combined_lower,
        positive=["other income", "tips", "gifts", "public assistance", "welfare"],
        negative=[],
    )
    # TANF check: only flag if explicitly disclosed as receiving TANF
    if not disclosures.has_other_income:
        if re.search(r"(?:receive|receiving|get|getting)\s+tanf", combined_lower):
            disclosures.has_other_income = True

    # --- Real estate ---
    disclosures.has_real_estate = _check_keyword(
        combined_lower,
        positive=["real estate", "own property", "own a home", "mortgage"],
        negative=["no real estate", "do not own"],
    )

    # --- Life insurance ---
    disclosures.has_life_insurance = _check_keyword(
        combined_lower,
        positive=["life insurance", "whole life"],
        negative=["no life insurance"],
    )

    return disclosures


def _check_employment(text: str, text_lower: str) -> tuple[bool | None, list[str]]:
    """Check for employment disclosures and extract employer names."""
    employers: list[str] = []

    # Interim Recertification Report patterns
    if "interim recertification report" in text_lower:
        # "A household member who was unemployed has gotten a job"
        if "gotten a job" in text_lower or "income has increased" in text_lower:
            # Extract employer name
            m = re.search(r"Employer'?s?\s*name.*?:\s*(.+?)(?:\n|Date|Address)", text, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if name and len(name) > 1:
                    employers.append(normalize_source_name(name) or name)

        # "I no longer work for this ..." pattern
        m = re.search(r"(?:no longer work|quit|left|terminated).*?(?:for|from|at)\s+(?:this\s+)?(.+?)(?:\s+since|\s+on|\n|$)", text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().rstrip(".")
            if name and len(name) > 1:
                employers.append(normalize_source_name(name) or name)

        if employers:
            return True, employers

    # General application patterns
    employer_patterns = [
        r"Employer[:\s]+(.+?)(?:\n|Address|Phone)",
        r"Place of (?:Employment|Work)[:\s]+(.+?)(?:\n|Address|Phone)",
        r"(?:Current|Present) Employer[:\s]+(.+?)(?:\n|Address|Phone)",
    ]
    for pattern in employer_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            name = m.group(1).strip()
            if name and len(name) > 1 and name.lower() not in ("none", "n/a", "unemployed"):
                employers.append(normalize_source_name(name) or name)

    if employers:
        return True, employers

    # Check for negative indicators
    if re.search(r"(?:unemployed|no employment|not employed|no income from employment)", text_lower):
        return False, []

    return None, []


def _check_keyword(
    text_lower: str,
    positive: list[str],
    negative: list[str],
) -> bool | None:
    """Check for positive/negative keyword matches.

    Returns True if positive match, False if negative match, None if neither.
    """
    # Check negative first (more specific)
    for kw in negative:
        if kw in text_lower:
            return False

    # Check positive
    for kw in positive:
        if kw in text_lower:
            return True

    return None
