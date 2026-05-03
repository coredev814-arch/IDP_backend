"""Compare IDP extraction against MuleSoft data from Salesforce.

Produces a list of structured Findings + a confidence score for the case.
Pure logic — no Salesforce calls, no I/O. Takes two dicts in, returns a result.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Finding categories
DUPLICATE_MEMBER = "DUPLICATE_MEMBER"
DUPLICATE_INCOME = "DUPLICATE_INCOME"
DUPLICATE_ASSET = "DUPLICATE_ASSET"
MISSING_MEMBER = "MISSING_MEMBER"     # AI found, MuleSoft missed
MISSING_INCOME = "MISSING_INCOME"     # AI found, MuleSoft missed
MISSING_ASSET = "MISSING_ASSET"       # AI found, MuleSoft missed
IDP_MISSED_MEMBER = "IDP_MISSED_MEMBER"   # MuleSoft has, AI missed
IDP_MISSED_INCOME = "IDP_MISSED_INCOME"
IDP_MISSED_ASSET = "IDP_MISSED_ASSET"
VALUE_MISMATCH = "VALUE_MISMATCH"
COMPLIANCE = "COMPLIANCE"


@dataclass
class Finding:
    category: str
    severity: str    # "high" | "medium" | "low" | "info"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Confidence:
    extraction_score: float
    agreement_rate: float
    case_confidence: float
    flag: str    # "green" | "yellow" | "red"
    agreements: int
    disagreements: int
    ai_only_records: int
    sf_only_records: int


@dataclass
class ComparisonResult:
    findings: list[Finding]
    confidence: Confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ssn_last4(ssn: str | None) -> str:
    if not ssn:
        return ""
    return str(ssn).replace("-", "").replace("*", "")[-4:]


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _money(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def _close(a: float, b: float, tolerance: float = 0.05) -> bool:
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) < tolerance


# ---------------------------------------------------------------------------
# Member comparison
# ---------------------------------------------------------------------------

def _member_key(rec: dict) -> tuple[str, str]:
    """Identity key: (DOB, last 4 of SSN). Stable across systems."""
    return (
        str(rec.get("DOB") or rec.get("DOB__c") or ""),
        _ssn_last4(rec.get("socialSecurityNumber") or rec.get("SSN__c")),
    )


def _compare_members(ai_members: list[dict], sf_members: list[dict]) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []

    # SF dedup: warn on duplicates within SF
    sf_keys: dict[tuple, list[dict]] = {}
    for m in sf_members:
        k = _member_key(m)
        sf_keys.setdefault(k, []).append(m)

    for k, group in sf_keys.items():
        if len(group) > 1 and k != ("", ""):
            name = f"{group[0].get('First_Name__c', '')} {group[0].get('Last_Name__c', '')}".strip()
            findings.append(Finding(
                category=DUPLICATE_MEMBER,
                severity="high",
                message=f"DUPLICATE MEMBER — {name} appears {len(group)} times in MuleSoft",
                detail={"key": list(k), "count": len(group)},
            ))

    ai_keys = {_member_key(m): m for m in ai_members if _member_key(m) != ("", "")}
    sf_keys_set = {k for k in sf_keys if k != ("", "")}

    matched = ai_keys.keys() & sf_keys_set
    ai_only = ai_keys.keys() - sf_keys_set
    sf_only = sf_keys_set - ai_keys.keys()

    for k in ai_only:
        m = ai_keys[k]
        name = f"{m.get('FirstName', '')} {m.get('LastName', '')}".strip()
        findings.append(Finding(
            category=MISSING_MEMBER,
            severity="high",
            message=(
                f"MuleSoft missing household member — {name} "
                f"(DOB: {m.get('DOB')}, SSN: ***-**-{k[1]})"
            ),
            detail={"name": name, "dob": m.get("DOB"), "ssn4": k[1]},
        ))

    for k in sf_only:
        m = sf_keys[k][0]
        name = f"{m.get('First_Name__c', '')} {m.get('Last_Name__c', '')}".strip()
        findings.append(Finding(
            category=IDP_MISSED_MEMBER,
            severity="medium",
            message=(
                f"AI may have missed household member — {name} "
                f"is in MuleSoft but not in AI extraction"
            ),
            detail={"name": name, "ssn4": k[1]},
        ))

    # Value mismatch on matched members (Disabled / Student)
    for k in matched:
        ai_m = ai_keys[k]
        sf_m = sf_keys[k][0]
        ai_dis = _norm(ai_m.get("disabled"))
        sf_dis = "y" if sf_m.get("Disabled__c") else "n" if sf_m.get("Disabled__c") is False else ""
        if ai_dis and sf_dis and ai_dis != sf_dis:
            findings.append(Finding(
                category=VALUE_MISMATCH,
                severity="medium",
                message=(
                    f"Disability mismatch — {ai_m.get('FirstName')} {ai_m.get('LastName')}: "
                    f"AI={ai_dis} MuleSoft={sf_dis}"
                ),
            ))

    return findings, {
        "agreements": len(matched),
        "disagreements": 0,
        "ai_only": len(ai_only),
        "sf_only": len(sf_only),
    }


# ---------------------------------------------------------------------------
# Income comparison
# ---------------------------------------------------------------------------

def _income_key(rec: dict) -> tuple[str, str]:
    """Source name + member name (normalized)."""
    return (
        _norm(rec.get("sourceName") or rec.get("Source_Name__c")),
        _norm(rec.get("memberName") or rec.get("House_Member_Name__c")),
    )


def _compare_income(
    ai_income: list[dict], ai_calculations: list[dict], sf_income: list[dict],
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []

    # Build map of AI annual amounts by source
    ai_annual: dict[tuple[str, str], float] = {}
    for calc in ai_calculations or []:
        k = (_norm(calc.get("sourceName")), _norm(calc.get("memberName")))
        amt = _money(calc.get("annualIncome"))
        if amt is not None:
            ai_annual[k] = amt

    # SF dedup check
    sf_keys: dict[tuple, list[dict]] = {}
    for inc in sf_income:
        k = _income_key(inc)
        sf_keys.setdefault(k, []).append(inc)

    # Detect SF duplicates that look like the same source with slight name diff
    sf_norm_sources = {k[0] for k in sf_keys if k[0]}
    for k, group in sf_keys.items():
        if len(group) > 1 and k[0]:
            findings.append(Finding(
                category=DUPLICATE_INCOME,
                severity="high",
                message=(
                    f"DUPLICATE INCOME — '{group[0].get('Source_Name__c')}' "
                    f"appears {len(group)} times in MuleSoft for the same member"
                ),
            ))

    ai_keys = set(_income_key(r) for r in ai_income)
    matched = ai_keys & set(sf_keys.keys())
    ai_only = ai_keys - set(sf_keys.keys())
    sf_only = set(sf_keys.keys()) - ai_keys

    agreements = 0
    disagreements = 0

    for k in matched:
        ai_amt = ai_annual.get(k)
        sf_amt = _money(sf_keys[k][0].get("Gross_Member_Income__c"))
        if ai_amt is not None and sf_amt is not None and sf_amt > 0:
            if _close(ai_amt, sf_amt, tolerance=0.01):
                agreements += 1
            else:
                disagreements += 1
                findings.append(Finding(
                    category=VALUE_MISMATCH,
                    severity="medium",
                    message=(
                        f"INCOME MISMATCH — {sf_keys[k][0].get('Source_Name__c')}: "
                        f"AI ${ai_amt:,.2f} vs MuleSoft ${sf_amt:,.2f}"
                    ),
                    detail={
                        "source": sf_keys[k][0].get("Source_Name__c"),
                        "ai_amount": ai_amt,
                        "sf_amount": sf_amt,
                    },
                ))
        else:
            agreements += 1     # both present, can't compare values

    for k in ai_only:
        findings.append(Finding(
            category=MISSING_INCOME,
            severity="high",
            message=f"MuleSoft missing income source — '{k[0]}' for {k[1] or 'household'}",
        ))

    for k in sf_only:
        findings.append(Finding(
            category=IDP_MISSED_INCOME,
            severity="medium",
            message=f"AI may have missed income source — '{k[0]}' is in MuleSoft",
        ))

    return findings, {
        "agreements": agreements,
        "disagreements": disagreements,
        "ai_only": len(ai_only),
        "sf_only": len(sf_only),
    }


# ---------------------------------------------------------------------------
# Asset comparison
# ---------------------------------------------------------------------------

def _asset_key(rec: dict) -> tuple[str, str, str]:
    """sourceName + accountType + last 4 of accountNumber."""
    src = _norm(rec.get("sourceName") or rec.get("Source_Name__c"))
    acct_type = _norm(rec.get("accountType") or rec.get("Account_Type__c"))
    acct_num = str(rec.get("accountNumber") or "").strip()
    if not acct_num:
        # Fall back to a balance-based stub for SF records without account #
        bal = (rec.get("Cash_Value__c") or rec.get("VOA_Current__c") or
               rec.get("currentBalance") or "")
        acct_num = f"~{bal}"
    last4 = acct_num[-4:].lower() if len(acct_num) >= 4 else acct_num.lower()
    return (src, acct_type, last4)


def _compare_assets(
    ai_assets: list[dict], sf_assets: list[dict],
) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []

    sf_keys: dict[tuple, list[dict]] = {}
    for a in sf_assets:
        k = _asset_key(a)
        sf_keys.setdefault(k, []).append(a)

    for k, group in sf_keys.items():
        if len(group) > 1:
            findings.append(Finding(
                category=DUPLICATE_ASSET,
                severity="high",
                message=(
                    f"DUPLICATE ASSET — '{group[0].get('Source_Name__c')}' "
                    f"({group[0].get('Account_Type__c')}) appears {len(group)} times"
                ),
            ))

    ai_keys = {_asset_key(a): a for a in ai_assets}
    matched = ai_keys.keys() & set(sf_keys.keys())
    ai_only = ai_keys.keys() - set(sf_keys.keys())
    sf_only = set(sf_keys.keys()) - ai_keys.keys()

    agreements = 0
    disagreements = 0

    for k in matched:
        ai_bal = _money(ai_keys[k].get("currentBalance"))
        sf_bal = _money(sf_keys[k][0].get("Cash_Value__c") or
                        sf_keys[k][0].get("VOA_Current__c"))
        if ai_bal is not None and sf_bal is not None:
            if _close(ai_bal, sf_bal, tolerance=0.02):
                agreements += 1
            else:
                disagreements += 1
                findings.append(Finding(
                    category=VALUE_MISMATCH,
                    severity="medium",
                    message=(
                        f"ASSET BALANCE MISMATCH — "
                        f"{sf_keys[k][0].get('Source_Name__c')} "
                        f"{sf_keys[k][0].get('Account_Type__c')}: "
                        f"AI ${ai_bal:,.2f} vs MuleSoft ${sf_bal:,.2f}"
                    ),
                ))
        else:
            agreements += 1

    for k in ai_only:
        rec = ai_keys[k]
        findings.append(Finding(
            category=MISSING_ASSET,
            severity="high",
            message=(
                f"MuleSoft missing asset — '{rec.get('sourceName')}' "
                f"({rec.get('accountType')}) ${_money(rec.get('currentBalance')) or 0:,.2f}"
            ),
        ))

    for k in sf_only:
        findings.append(Finding(
            category=IDP_MISSED_ASSET,
            severity="medium",
            message=(
                f"AI may have missed asset — '{sf_keys[k][0].get('Source_Name__c')}' "
                f"is in MuleSoft"
            ),
        ))

    return findings, {
        "agreements": agreements,
        "disagreements": disagreements,
        "ai_only": len(ai_only),
        "sf_only": len(sf_only),
    }


# ---------------------------------------------------------------------------
# Compliance findings (IDP-internal — no MuleSoft comparison needed)
# ---------------------------------------------------------------------------

def _compliance_findings(extraction: dict) -> list[Finding]:
    findings: list[Finding] = []
    cert = extraction.get("certification_info") or {}
    if str(cert.get("isSigned") or "").lower() == "no":
        findings.append(Finding(
            category=COMPLIANCE,
            severity="high",
            message="UNSIGNED TIC — TIC form not signed. Resubmission required.",
        ))
    missing = cert.get("missingForms") or []
    for m in missing:
        findings.append(Finding(
            category=COMPLIANCE,
            severity="medium",
            message=f"MISSING FORM — {m}",
        ))
    return findings


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def _calculate_confidence(
    extraction: dict,
    member_stats: dict, income_stats: dict, asset_stats: dict,
) -> Confidence:
    field_scores = extraction.get("field_scores") or {}
    extraction_score = float(field_scores.get("overall_composite") or 0.0)

    agreements = (member_stats["agreements"] + income_stats["agreements"]
                  + asset_stats["agreements"])
    disagreements = (member_stats["disagreements"] + income_stats["disagreements"]
                     + asset_stats["disagreements"])
    ai_only = (member_stats["ai_only"] + income_stats["ai_only"]
               + asset_stats["ai_only"])
    sf_only = (member_stats["sf_only"] + income_stats["sf_only"]
               + asset_stats["sf_only"])

    # Don't penalize "AI catches MuleSoft miss"; do penalize disagreements
    # and "AI missed something MuleSoft has".
    weight = agreements + disagreements + (sf_only * 0.5)
    if weight > 0:
        rate = (agreements - 0.5 * sf_only) / weight
        agreement_rate = max(0.0, min(1.0, rate))
    else:
        # No comparable records (MuleSoft has nothing) — confidence
        # rests on extraction quality alone.
        agreement_rate = 1.0 if ai_only > 0 else 0.5

    case_confidence = (extraction_score * 0.6) + (agreement_rate * 0.4)
    flag = ("green" if case_confidence >= 0.85
            else "yellow" if case_confidence >= 0.65
            else "red")

    return Confidence(
        extraction_score=extraction_score,
        agreement_rate=agreement_rate,
        case_confidence=case_confidence,
        flag=flag,
        agreements=agreements,
        disagreements=disagreements,
        ai_only_records=ai_only,
        sf_only_records=sf_only,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compare(extraction: dict, sf_data: dict) -> ComparisonResult:
    """Diff IDP extraction against MuleSoft data; return findings + score."""
    ai_members = (extraction.get("household_demographics") or {}).get("houseHold") or []
    ai_income = ((extraction.get("income") or {}).get("sourceIncome") or {}).get("verificationIncome") or []
    ai_calculations = extraction.get("income_calculations") or []
    ai_assets = (extraction.get("assets") or {}).get("assetInformation") or []

    sf_members = sf_data.get("members") or []
    sf_income = sf_data.get("income") or []
    sf_assets = sf_data.get("assets") or []

    member_findings, member_stats = _compare_members(ai_members, sf_members)
    income_findings, income_stats = _compare_income(ai_income, ai_calculations, sf_income)
    asset_findings, asset_stats = _compare_assets(ai_assets, sf_assets)
    compliance_findings = _compliance_findings(extraction)

    findings = (
        member_findings + income_findings + asset_findings + compliance_findings
    )
    confidence = _calculate_confidence(
        extraction, member_stats, income_stats, asset_stats,
    )

    return ComparisonResult(findings=findings, confidence=confidence)
