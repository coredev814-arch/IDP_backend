"""Field extraction service — uses LLM to extract structured data per MuleSoft schema."""

import logging

from app.core.config import Settings
from app.schemas.extraction import (
    AssetExtraction,
    CertificationInfo,
    DocumentGroup,
    DocumentInventory,
    HouseholdDemographics,
    IncomeExtraction,
)
from app.services.llm_service import call_llm_json
from app.services import validation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cert-type context injected into every extraction prompt (Section 12)
# ---------------------------------------------------------------------------

_CERT_TYPE_CONTEXT = {
    "MI": (
        "\n\nCERTIFICATION TYPE: MI (Move-In / Initial Certification)\n"
        "- Extract ALL income sources, assets, and household composition from scratch.\n"
        "- TIC/HUD 50059 is reference only — source documents (VOI, paystubs, bank statements) are the primary source of truth.\n"
        "- Self-declared amounts come from the Application for Housing.\n"
        "- There is NO previous certification to compare against."
    ),
    "AR": (
        "\n\nCERTIFICATION TYPE: AR (Annual Recertification)\n"
        "- Extract ALL current income sources and assets — full verification required.\n"
        "- TIC/HUD 50059 is reference only — source documents are the primary source of truth.\n"
        "- Self-declared amounts come from the Recertification Questionnaire.\n"
        "- A previous certification may exist for comparison but do NOT extract from it."
    ),
    "AR-SC": (
        "\n\nCERTIFICATION TYPE: AR-SC (Annual Recertification — Self-Certification)\n"
        "- CRITICAL: The TIC form IS the source of truth for all income and asset data.\n"
        "- Values from the TIC can be used as self-declared amounts on Income and Asset Worksheets.\n"
        "- Full third-party verification is NOT required — tenant self-certifies.\n"
        "- If TIC shows income amounts, use those as selfDeclaredAmount with selfDeclaredSource='Self-Certification TIC'."
    ),
    "IR": (
        "\n\nCERTIFICATION TYPE: IR (Interim Recertification)\n"
        "- Triggered by a CHANGE in income or household composition.\n"
        "- Focus on extracting the CHANGED income source(s), not re-extracting everything.\n"
        "- The Recertification Questionnaire/Report indicates what changed.\n"
        "- A previous certification exists — do NOT extract from previous cert pages.\n"
        "- TIC/HUD 50059 is reference only — source documents are the primary source of truth."
    ),
}


def _get_cert_context(certification_type: str | None) -> str:
    """Get cert-type context string for injection into LLM prompts."""
    if not certification_type:
        return ""
    return _CERT_TYPE_CONTEXT.get(certification_type.upper(), "")

# ---------------------------------------------------------------------------
# System prompts — one per MuleSoft schema
# ---------------------------------------------------------------------------

DEMOGRAPHICS_SYSTEM_PROMPT = """\
You are an expert data extractor for HUD/Affordable Housing certification documents.

Extract household member demographics from the provided document text.

CRITICAL RULES:
- SSN MASKING: NEVER output a full SSN. ALWAYS mask as ***-**-XXXX (last 4 digits only).
- NAME FORMATTING: ALWAYS Title Case for ALL names.
- If the document is a Calculation Worksheet (keywords: "Calculation", "Calculator", "Worksheet", "Calc Sheet", "PCAP", "CF-51", "LIHTC Calc"), return {"houseHold": []} immediately.

EXTRACTION RULES:
- householdMemberNumber: 2-digit string with leading zero ("01", "02"). From TIC → "HH Mbr#"; HUD 50059 → Field 33 "No."; RD 3560 → derive from row order. null for pay stubs, bank statements, etc.
- FirstName: Title Case. null if only single name field.
- MiddleName: Title Case, or null.
- LastName: Title Case. Include suffixes (Jr., Sr., III). Preserve hyphens and multi-word names.
- socialSecurityNumber: ALWAYS ***-**-XXXX. null if not found.
- DOB: YYYY-MM-DD format. "01/15/1990" → "1990-01-15". If no day, default DD to 01.
- head: "H" for head of household/primary applicant. null for all others. Maximum ONE "H".
- disabled: "Y" if member is disabled, "N" if not disabled, null if unknown/not documented.
  HUD 50059 has MULTIPLE disability indicators:
  (a) Per-member: Section C column "Special Status" or "Disab" or "H/C" — check marks, "Y", "1", "X" = "Y"; blank = "N"
  (b) Family-level: Fields like "Family has Mobility Disability? N", "Family has Hearing Disability? N", "Family has Visual Disability? N" — if ALL are "N", set disabled="N" for ALL members. If ANY is "Y", set disabled="Y" for the head of household (member 1) unless a specific member is identified.
  (c) On TIC: look in HOUSEHOLD COMPOSITION for "Disability" or "Handicapped" column.
  IMPORTANT: If disability fields exist anywhere in the document (even as family-level fields), you MUST set disabled to "Y" or "N" for each member. Only set to null if NO disability information exists at all.
- student: "Y" if member is a student, "N" if not, null if unknown/not documented.
  HUD 50059: Section C column "Stdnt" or "FT Student" — check marks, "Y", "1", "X" = "Y"; blank column with no marks = "N" for all members.
  On TIC: look in HOUSEHOLD COMPOSITION for "F/T Student" or "Student" column.
  IMPORTANT: If a student column EXISTS in the document (even if all blank/empty), set student="N" for all members. Only set to null if no student column exists at all.
- email: Exact as shown. null if no valid email.
- phone: Normalize to (XXX) XXX-XXXX. null if not 10 digits.

DEDUPLICATION: Each unique member appears only once. Match by: member number (primary), SSN last 4 (secondary), FirstName+LastName (tertiary).

Return ONLY valid JSON: {"houseHold": [...]}"""

CERT_INFO_SYSTEM_PROMPT = """\
You are an expert data extractor for HUD/Affordable Housing certification documents.

Extract certification-level information from TIC forms, HUD 50059 forms, HUD 3560 forms,
or HUD Model Lease agreements. Only extract from the CURRENT certification — ignore any
document classified as "(Previous)".

CRITICAL RULES:
- If the document is a Calculation Worksheet, return {"certificationInfo": {}} immediately.
- If the document is a previous certification, return {"certificationInfo": {}} immediately.
- A HUD Model Lease is a valid source for effectiveDate, grossRent, tenantRent,
  utilityAllowance, unitNumber, and signatureDate even if the cert form itself is missing
  those fields. The lease uses plain-English numbered paragraphs, not form field codes.

ANTI-HALLUCINATION GUARD (CRITICAL):
- If a numeric field is BLANK on the form — empty line, "$" with no number,
  "$0" / "$0.00" / "0.00", dashes, "N/A", or literally nothing next to the
  label — return null for that field. Do NOT invent a plausible value.
- Every extracted value must be a literal string that appears in the document
  text. Before outputting a number, verify the exact digits are present.
- If income shows $0 across ALL sources (TIC Part III totals = $0, HUD 50059
  field 86 = $0, RD 3560-8 Line 18.f = $0), then householdIncome = "0.00"
  (not null, not a guess). Zero-income households are legitimate.
- For rent fields: if the primary form is blank, check secondary forms per
  the MULTI-SOURCE FALLBACK rules. If ALL are blank, return null.
- NEVER map unrelated numbers (e.g., security deposit, utility schedule,
  passbook rate, field numbers like "30" or "31") to rent fields.

FIELDS TO EXTRACT:
- certificationType: Certification type code. Values: "MI" (Move-In/Initial), "AR" (Annual Recertification), "AR-SC" (Annual Recert Self-Certification), "IR" (Interim Recertification). Look for: "Type of Certification" field, checkboxes for Initial/Annual/Interim, or coded fields on the form.
- effectiveDate: Effective date of the certification. YYYY-MM-DD format.
- numberOfBedrooms: Number of bedrooms. Numeric string.
- grossRent: Total tenant payment or gross rent amount. Numeric string with 2 decimals, no $ or commas.
- tenantRent: Tenant rent portion. Numeric string with 2 decimals.
- utilityAllowance: Utility allowance amount. Numeric string with 2 decimals.
- rentLimit: Rent limit for the unit. Numeric string with 2 decimals.
- householdIncome: Total annual household income. Numeric string with 2 decimals.
- householdSize: Number of household members. Integer as string.
- unitNumber: Unit number or apartment number.
- signatureDate: Date the form was signed. YYYY-MM-DD.
- isSigned: "Yes" if signature is present, "No" if signature line is blank.
- applicationSignDate: Application sign date if present. YYYY-MM-DD.

DOCUMENT-SPECIFIC GUIDANCE:
- TIC Form: Cert type is in the header area (checkboxes for Initial/Annual/Interim/Other). Effective date is labeled "Effective Date". Income is in Part III "Income". Rent fields are in Part IV "Rent".
- HUD 50059: Cert type is field 2b "Type of Action" (1=Initial, 2=Annual, 3=Interim, etc.). Effective date is field 2a. Field 29 = Contract Rent, Field 30 = Utility Allowance, Field 31 = Gross Rent (this is the true grossRent, NOT field 29). Field 110 = Tenant Rent. Field 86 = Total Annual Income.
- HUD 3560 (RD 3560-8 / USDA): Line 30.a = Note Rate Rent (use as tenantRent or grossRent depending on form), Line 30.b = Utility Allowance, Line 30.c = Gross Note Rate Rent (use as grossRent). Line 33 = Final NTC (Net Tenant Contribution = tenantRent). Line 18.f = Monthly Income, Line 20 = Adjusted Annual Income.
- HUD Model Lease: Gross Rent = Contract Rent + Utility Allowance. Record grossRent from the "Gross Rent" line if shown, otherwise compute Contract Rent + UA. "Tenant Rent" / tenant's portion = tenantRent. "Utility Allowance" = utilityAllowance. "Unit" / dwelling unit number = unitNumber. Lease commencement date = effectiveDate. Signature date on the lease = signatureDate.

MULTI-SOURCE FALLBACK (CRITICAL):
When the primary certification form (TIC / HUD 50059 / HUD 3560) has a BLANK
field, look for the same data on a secondary form within the same group:
  1. TIC rent fields blank → check RD 3560-8 Line 30.a/b/c (on same cert)
  2. TIC income blank → check household income on RD 3560-8 Line 18.f × 12 or Line 20
  3. HUD 50059 fields missing → check attached HUD Model Lease / Notice of Rent Change
  4. Any primary form missing effectiveDate → fall back to:
     - TIC Part X "Date Signed" or Part VII "Move-in Date"
     - HUD 50059 owner signature date
     - HUD Model Lease commencement date
     - Notice of Rent Change "effective with the rent due for"
NEVER invent a rent figure that is not present somewhere in the document text.
If all sources are blank, return null for that field.

Return ONLY valid JSON: {"certificationInfo": {...}}"""

INCOME_SYSTEM_PROMPT = """\
You are an expert data extractor for HUD/Affordable Housing income documents.

Extract income data from the provided document text into the MuleSoft Income schema.

CRITICAL RULES:
- SSN MASKING: ALWAYS ***-**-XXXX. No exceptions.
- NAME FORMATTING: ALWAYS Title Case.
- If document is a Calculation Worksheet, return {"sourceIncome": {"payStub": [], "verificationIncome": []}} immediately.

DOCUMENT ROUTING:
- payStub route: Pay stubs, pay-slips, Work Number/Equifax/ScreeningWorks/Vault Verify wage records
- verificationIncome route: SSA/EIV benefit letters, TANF, child support, cash contributions, pension, self-employment, sworn statements, VOI/VOE

CHILD SUPPORT STATEMENT — ALWAYS EXTRACT:
Any "Child Support Statement", "Child Support Order", "Child Support Verification",
court order with ordered amounts, or DOR/State Disbursement Unit statement → create
a verificationIncome entry:
  - sourceName: payer name, or "Child Support" / state agency name if payer unknown
  - memberName: the custodial parent/head of household receiving support
  - incomeType: "Child Support"
  - type_of_VOI: "Child Support Order"
  - selfDeclaredAmount: monthly or annual support amount as shown
  - rateOfPay: support amount per payment period, if shown
  - frequencyOfPay: payment frequency (weekly, bi-weekly, monthly), if shown
Do NOT skip child support just because the form is brief or lacks typical wage fields.
If a TIC or cert form lists household income that exceeds the sum of wage sources, and
a Child Support Statement is present, the gap is almost always the child support amount.

PAYSTUB FIELDS:
- sourceName: employer name, Title Case
- memberName: employee name, Title Case
- socialSecurityNumber: ***-**-XXXX
- grossPay: exact dollar amount with cents, numeric string ("1250.00"), no $ or commas
- payDate: YYYY-MM-DD
- payInterval: lowercase (weekly / bi-weekly / semi-monthly / monthly)

SPECIAL PAYSTUB RULES:
- Work Number/Equifax: take the 6 most current entries only
- Child Support: last 6 most recent payments
- Do NOT create pay stubs for SSA, pension, or TANF (these go to verificationIncome)

VERIFICATION INCOME FIELDS:
- sourceName, memberName, socialSecurityNumber (same rules as payStub)
- programName: official program name for benefit income
- selfDeclaredAmount: from self-cert forms, applications, or questionnaires, numeric string. Match to the corresponding employer/source by name when possible.
- dateReceived: date the VOI form was received or date signed by employer. YYYY-MM-DD. null if not shown.
- rateOfPay: numeric string (hourly or periodic rate)
- frequencyOfPay: lowercase. This is how often the person is PAID (weekly / bi-weekly / semi-monthly / monthly), NOT the rate unit. If rate is "hourly" but pay dates are 14 days apart, frequencyOfPay is "bi-weekly". Determine from pay period structure, not from rate label.
- hoursPerPayPeriod: hours per week
- overtimeRate: only if person actually receives overtime
- overtimeFrequency: same frequency as regular pay
- ytdAmount: only if document explicitly states "year to date". MUST BE null for SSA/fixed income.
- ytdStartDate, ytdEndDate: YYYY-MM-DD
- incomeType: one of: Non-Federal Wage, Federal Wage, Social Security, Supplemental Security Income, Social Security Disability, Pension, Temporary Assistance, Child Support, Self-Employment, Zero Income, Other Income
- type_of_VOI: Employer Verification, SSA Benefit Letter, Agency Benefit Letter, Child Support Order, Pension Statement, Self-Declaration, Work Number, ScreeningWorks, Vault Verify
- address: {street, city, state (2-letter), zip (5-digit)} or null
- employmentStatus: "Active" if currently employed, "Terminated" if employment has ended, "On Leave" if on leave. Extract from "Presently Employed" checkbox or employment status field. This is CRITICAL for understanding the income picture.
- terminationDate: YYYY-MM-DD. Extract if employment has ended (last day worked, termination date, or separation date).
- hireDate: YYYY-MM-DD. Extract from "Date Hired", "Start Date", "Original Hire Date", or "Date of Hire".

*** EQUIFAX / WORK NUMBER SPECIAL RULES (CRITICAL — OVERRIDE DEFAULTS) ***:
- ytdStartDate: MUST be the "Original Hire Date" or "Most Recent Start Date" shown on the Equifax report. NEVER use January 1 or any calendar year start. Example: if report says "Original Hire Date: 02/13/2026", then ytdStartDate = "2026-02-13".
- ytdEndDate: MUST be the report's "Current As Of" date or "Inquiry Date". Example: if "Current As Of: 03/03/2026" or "Inquiry Date: 03/10/2026", use whichever is the later date.
- frequencyOfPay: MUST be determined from the "Pay Cycle" field or pay period dates, NOT from the rate label. "Pay Cycle: Biweekly" → frequencyOfPay = "bi-weekly". "Pay Frequency: Hourly" is the RATE unit (goes in rateOfPay), not the pay frequency. If "Pay Cycle" says "Biweekly" and rate says "$18.00 Hourly", then rateOfPay = "18.00" and frequencyOfPay = "bi-weekly".
- hireDate: Extract from "Original Hire Date" or "Most Recent Start Date".

SELF-DECLARED AMOUNTS FROM QUESTIONNAIRE:
- When an Application or Housing Questionnaire mentions income amounts (e.g., "income has changed", "currently earning", "expected income"), extract as selfDeclaredAmount on the matching verificationIncome entry.
- Match self-declared amounts to the corresponding employer/source by name.
- If a questionnaire states a specific dollar amount for an employer, place it in selfDeclaredAmount of that employer's verificationIncome entry.

FORMATTING:
- Monetary: numeric string with 2 decimal places, no $ or commas
- Dates: YYYY-MM-DD
- Names: Title Case
- payInterval/frequencyOfPay: lowercase

Return ONLY valid JSON: {"sourceIncome": {"payStub": [...], "verificationIncome": [...]}}"""

ASSET_SYSTEM_PROMPT = """\
You are an expert data extractor for HUD/Affordable Housing asset documents.

Extract asset data from the provided document text into the MuleSoft Asset schema.

CRITICAL RULES:
- SSN MASKING: ALWAYS ***-**-XXXX.
- NAME FORMATTING: ALWAYS Title Case.
- If document is a Calculation Worksheet, return {"assetInformation": []} immediately.

DOCUMENT ROUTING:
- bankStatment route: actual bank statements with transactions
- verificationOfAsset route: VOA/VOD forms, life insurance, investment statements
- Self-declared amounts: from asset self-certification, applications, questionnaires

FIELDS:
- documentType: Bank Statement — Checking, Bank Statement — Savings, Verification of Assets, Life Insurance, Investment Account, Asset Self-Certification, ABLE Account, Real Estate, Certificate of Deposit, Cryptocurrency, Prepaid Card, Annuity, Direct Express Card
- assetOwner: account holder name, Title Case
- socialSecurityNumber: ***-**-XXXX
- sourceName: institution name, Title Case
- selfDeclaredAmount: from self-cert forms only
- accountType: Checking, Savings, CD, Investment, Retirement, Life Insurance, Cryptocurrency, Prepaid Card, Peer-to-Peer, ABLE Account, Real Estate, Cash, Annuity, Direct Express
- accountNumber: as shown on document
- currentBalance, averageSixMonthBalance: numeric string, 2 decimals
- dateReceived: YYYY-MM-DD
- incomeAmount: income from asset
- interestType: "Dollar Amount" or "Percentage"
- percentageOfOwnership: numeric string (e.g., "100", "50")
- address: {street, city, state, zip} or null

NESTED OBJECTS:
- bankStatment: array of bank statement entries. Each entry has:
  {statementDate, balance, accountNumber, currentMortgageBalance, income, incomeFixedValue, incomeFromAsset, interestRate, netValueRealEstate, percentageOfOwnership, realEstateCurrentMarketValue, totalClosingCosts}
  All monetary fields are numeric strings with 2 decimals. Empty [] if no statements.
- verificationOfAsset: {accountNumber, currentBalance, averageSixMonthBalance, dateReceived, incomeAmount, interestType, interestRate, percentageOfOwnership}. null if no VOA.

SPECIAL RULES:
- Life insurance: ALWAYS use cash/surrender value, NEVER use face value. If only face value is shown, set currentBalance to null and add note "Only face value available — cash value not provided"
- Thomson Reuters / WestlawNext VOA forms: treat as Verification of Assets. Extract per account: account number, account type (checking/savings), account balance, average balance, date received
- Joint/shared accounts: capture percentageOfOwnership. If ownership is split (e.g., 50% with non-household member), record the percentage
- Each distinct account = separate array entry
- Do NOT extract from manager worksheets (Asset Self-Certification — Manager's Worksheet)

HUD 50059 / TIC ASSET EXTRACTION:
- HUD 50059 Section D contains asset information (fields 76-80): Description, Status, Cash Value, Actual Yearly Income, Date Divested
- Extract each asset row as a separate entry. Map "Cash Value" → currentBalance, "Actual Yearly Income" → incomeAmount
- Common descriptions: "Checking account", "Savings account", "Other asset", "Life insurance", etc.
- The member number column tells you which household member owns the asset — match to person name from Section C
- TIC Part V contains similar asset data — extract the same way

Return ONLY valid JSON: {"assetInformation": [...]}"""

INVENTORY_FINANCIAL_SYSTEM_PROMPT = """\
You are an expert document cataloger for HUD/Affordable Housing certification files.

Catalog all FINANCIAL and COMPLIANCE documents (NOT HUD-specific forms) found in the provided text.

DOCUMENT TYPES TO CATALOG (23 types):
Pay Stub, Bank Statement, SSA Benefit Letter, SSI Benefit Letter, SSDI Benefit Letter, Pension Statement / Letter, TANF / Public Assistance Verification, Child Support Order / Statement, Verification of Income (VOI / VOE), Verification of Deposit (VOD), Verification of Assets (VOA), Life Insurance Policy / Statement, Asset Self-Certification, Divestiture of Assets Certification, Student Status Affidavit / Certification, Annual Student Certification, Unemployed / Zero Income Affidavit, Child Support / Alimony Affidavit, Household Demographics Form, Application / Household Roster, Self-Declaration / Self-Certification, Work Number / Equifax Income Report, Income Calculation Worksheet (flag only)

SKIP THESE (handled by HUD inventory):
HUD 9887, HUD 9887-A, HUD 92006, EIV Summary Report, EIV Income Report, Acknowledgement of Receipt of HUD Forms, Race and Ethnic Data Reporting Form, Authorization to Release Information, Tenant Release and Consent Form

EXCLUDE ENTIRELY:
Fax cover sheets, blank/unfilled forms, credit screening reports, IRS Form 4506-T, duplicate pages, internal manager correspondence

FIELDS:
- documentType: from the 23 types above
- documentTitle: specific title/heading on the document
- sourceOrganization: issuing agency/employer/bank, Title Case
- personName: person named on document, Title Case
- pageRange: e.g. "1", "4-5", "7-9"
- pageCount: number of pages
- isSigned: "Yes", "No", or "N/A"
- signedBy: signer name in Title Case, or null
- signatureDate: YYYY-MM-DD or null
- documentDate: YYYY-MM-DD or null
- notes: notable observations or null

RULES:
- NO PII extraction (no SSNs, account numbers, financial amounts)
- Consecutive pages of same doc + same person = single entry
- Same doc type for different people = separate entries
- Income Calculation Worksheets: include with note "Internal calculation document — excluded from IDP data extraction per Document Exclusion rules"
- Always catalog: Asset Self-Cert, Student Status, Zero Income, Application, etc.
- If poor scan quality, note it

Return ONLY valid JSON: {"documents": [...]}"""

INVENTORY_HUD_SYSTEM_PROMPT = """\
You are an expert document cataloger for HUD compliance forms in affordable housing certification files.

Catalog ONLY the following 9 HUD compliance form types. Ignore ALL other documents.

HUD FORMS TO CATALOG:
1. Authorization to Release Information — property-specific, authorizes income/asset verification
2. Tenant Release and Consent — authorizes release of tenant info to third parties
3. HUD-9887 — "Notice and Consent for Release of Information", ONE per household, signed by ALL adults
4. HUD-9887-A — "Applicant's/Tenant's Consent", ONE per adult member, signed by tenant AND owner
5. HUD-92006 — "Supplement to Application for Federally Assisted Housing", emergency contact info
6. EIV Summary Report — system-generated, household-level summary, NOT signed
7. EIV Income Report — system-generated, wage/benefit data, NOT signed
8. Acknowledgement of Receipt of HUD Forms — checklist of received documents, signed by resident
9. Race and Ethnic Data Reporting Form — OMB No. 2502-0204, ONE per household member, race/ethnicity checkboxes

FIELDS:
- documentType: one of the 9 types above
- documentTitle: specific title on the form
- sourceOrganization: property management company or HUD, Title Case
- personName: person named on form, Title Case
- pageRange: e.g. "1", "4-5"
- pageCount: number of pages
- isSigned: "Yes", "No", or "N/A" (N/A for system-generated EIV reports)
- signedBy: signer name in Title Case, or null
- signatureDate: YYYY-MM-DD or null
- documentDate: YYYY-MM-DD or null
- notes: notable observations or null

RULES:
- NO PII extraction
- Same form for different people = separate entries
- Multi-page forms = single entry with full page range
- If zero HUD forms found, return {"documents": []}
- Note poor scan quality if applicable

Return ONLY valid JSON: {"documents": [...]}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_texts(groups: list[DocumentGroup]) -> list[str]:
    """Build LLM input texts from pre-routed groups. No filtering — trust the router."""
    texts = []
    for g in groups:
        if g.category == "ignore":
            continue
        texts.append(
            f"[Document: {g.document_type}, Pages: {g.page_range}, "
            f"Person: {g.person_name or 'Unknown'}]\n{g.combined_text}"
        )
    return texts


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_demographics(
    groups: list[DocumentGroup],
    settings: Settings,
    certification_type: str | None = None,
) -> HouseholdDemographics:
    """Extract household demographics from pre-routed document groups."""
    relevant_texts = _build_texts(groups)
    if not relevant_texts:
        logger.info("No demographic documents found")
        return HouseholdDemographics()

    user_prompt = (
        "Extract household member demographics from these documents:\n\n"
        + "\n\n---\n\n".join(relevant_texts)
    )
    user_prompt += _get_cert_context(certification_type)

    result = call_llm_json(DEMOGRAPHICS_SYSTEM_PROMPT, user_prompt, settings)
    result = validation.validate_household(result)

    members = result.get("houseHold", [])
    logger.info("Extracted %d household members", len(members))
    return HouseholdDemographics.model_validate(result)


def extract_certification_info(
    groups: list[DocumentGroup],
    settings: Settings,
    certification_type: str | None = None,
) -> CertificationInfo:
    """Extract certification-level info from pre-routed cert form groups.

    After the first extraction, checks for critical fields that came back
    null and issues ONE targeted retry for just those fields. LLM extraction
    is non-deterministic — the same prompt on the same document can skip
    fields on one run and populate them on the next. A focused retry
    ("extract ONLY these 4 fields") usually recovers them because the model
    isn't juggling 10 fields at once.

    Retry policy:
      - Trigger: any of the _CRITICAL_FIELDS is null after the first pass
      - Scope: re-ask only for the null fields (constrained prompt)
      - Merge: retry values fill nulls; already-populated fields are kept
      - Max retries: 1 (no loop)
    """
    relevant_texts = _build_texts(groups)
    if not relevant_texts:
        logger.info("No certification documents found")
        return CertificationInfo()

    user_prompt = (
        "Extract certification information from these documents:\n\n"
        + "\n\n---\n\n".join(relevant_texts)
    )
    user_prompt += _get_cert_context(certification_type)

    result = call_llm_json(CERT_INFO_SYSTEM_PROMPT, user_prompt, settings)
    result = validation.validate_certification_info(result)

    cert_info_dict = result.get("certificationInfo", {}) or {}

    # --- Targeted retry for missing critical fields ---
    missing = [f for f in _CRITICAL_CERT_FIELDS if not cert_info_dict.get(f)]
    if missing:
        logger.info(
            "Cert info partial extraction — retrying for %d missing fields: %s",
            len(missing), missing,
        )
        retry_dict = _retry_cert_info_fields(
            relevant_texts, missing, cert_info_dict, certification_type, settings,
        )
        # Merge: retry values fill nulls only, never overwrite populated fields
        for field in missing:
            retry_val = retry_dict.get(field)
            if retry_val not in (None, "", "null"):
                cert_info_dict[field] = retry_val
        recovered = [f for f in missing if cert_info_dict.get(f)]
        if recovered:
            logger.info("Retry recovered %d/%d fields: %s",
                        len(recovered), len(missing), recovered)

    # Cert type is caller-provided (frontend upload form / API param).
    # Overwrite any LLM guess with the authoritative value.
    if certification_type:
        cert_info_dict["certificationType"] = certification_type

    logger.info(
        "Extracted certification info: type=%s",
        cert_info_dict.get("certificationType"),
    )
    return CertificationInfo.model_validate(cert_info_dict)


# Fields considered critical for cert_info. If any of these come back null
# from the first extraction, a targeted retry runs to try to recover them.
# These are the fields that downstream scoring, compliance checks, and the
# frontend display all depend on.
_CRITICAL_CERT_FIELDS = [
    "effectiveDate",
    "grossRent",
    "tenantRent",
    "utilityAllowance",
    "householdIncome",
    "unitNumber",
    "householdSize",
    "numberOfBedrooms",
]


_CERT_INFO_RETRY_PROMPT = """\
You are re-examining a HUD/Affordable Housing certification form to extract
specific fields that were missed on the first pass.

Extract ONLY the fields listed below. Return a JSON object containing just
those field names as keys. If a field is genuinely not present in the
document (e.g., the form line is blank), return null for that key — but try
hard first: the field is almost certainly in the text somewhere.

Field meanings:
- effectiveDate: Effective date of this certification (YYYY-MM-DD). Check, in order:
  (1) TIC/HUD 50059/RD 3560 "Effective Date" field,
  (2) TIC Part X "Date Signed" or Part VII "Move-in Date",
  (3) HUD 50059 owner signature date,
  (4) HUD Model Lease commencement date,
  (5) Notice of Rent Change / Lease Amendment "effective with the rent due for [date]".
- grossRent: Gross rent amount (numeric, 2 decimals, no $ or commas). For HUD
  50059 this is FIELD 31, NOT field 29. For RD 3560-8 this is Line 30.c.
  If blank on the primary form, check RD 3560-8 / Model Lease / Notice of Rent Change.
- tenantRent: Tenant's portion of rent (numeric). HUD 50059 field 110, TIC Part IV
  "Tenant Rent", RD 3560-8 Line 33 (Final NTC). If blank, check lease or notice.
- utilityAllowance: Utility allowance amount (numeric). HUD 50059 field 30, RD 3560-8
  Line 30.b, TIC Part IV "Utility Allowance".
- householdIncome: Total annual household income (numeric). HUD 50059 field 86, TIC
  Part III "Total Income (E)", RD 3560-8 Line 18.f × 12 or Line 20.
- unitNumber: Unit number or apartment number. Strip any building prefix
  (e.g., "Bldg 2 Unit 27" → "27", "2 27" → "27").
- householdSize: Number of household members (integer as string)
- numberOfBedrooms: Number of bedrooms (numeric string)

Return ONLY valid JSON: {"field_name": value, ...}
Do NOT include fields not in the missing list.

ANTI-HALLUCINATION GUARD (CRITICAL):
- Every extracted number must be literal text in the document. If the field
  is blank (empty, "$" with no number, "$0", "N/A"), return null.
- Do NOT invent plausible values. Better to return null than to guess.
- Do NOT map field numbers ("30", "31", "86"), line numbers, percentages,
  or unrelated figures (security deposit, passbook rate) to rent fields.
- For zero-income households where all income sources show $0, return
  "0.00" for householdIncome — not a guess, not null."""


def _retry_cert_info_fields(
    relevant_texts: list[str],
    missing_fields: list[str],
    already_extracted: dict,
    certification_type: str | None,
    settings: Settings,
) -> dict:
    """Targeted retry for specific missing cert_info fields.

    Sends a narrower prompt asking for ONLY the missing fields, along with
    the already-extracted values as context so the model can cross-reference.
    """
    from app.services.llm_service import call_llm_json as _call

    # Show already-extracted values so the retry has context but knows not
    # to overwrite them.
    context_lines = [
        f"  {k}: {v}" for k, v in already_extracted.items()
        if v not in (None, "", "null") and k not in missing_fields
    ]
    context_block = (
        "\nAlready extracted (for context, do NOT re-extract these):\n"
        + "\n".join(context_lines) if context_lines else ""
    )

    user_prompt = (
        f"Missing fields to extract: {', '.join(missing_fields)}\n"
        f"{context_block}\n\n"
        f"DOCUMENT TEXT:\n\n"
        + "\n\n---\n\n".join(relevant_texts)
    )
    user_prompt += _get_cert_context(certification_type)

    try:
        result = _call(_CERT_INFO_RETRY_PROMPT, user_prompt, settings)
    except Exception:
        logger.exception("Cert info retry call failed — keeping original values")
        return {}

    # Accept either {"field": val} or {"certificationInfo": {"field": val}}
    if isinstance(result, dict) and "certificationInfo" in result:
        result = result.get("certificationInfo") or {}
    return result if isinstance(result, dict) else {}


def extract_income(
    groups: list[DocumentGroup],
    settings: Settings,
    certification_type: str | None = None,
) -> IncomeExtraction:
    """Extract income data from pre-routed income document groups."""
    relevant_texts = _build_texts(groups)
    if not relevant_texts:
        logger.info("No income documents found")
        return IncomeExtraction()

    user_prompt = "Extract income data from these documents:\n\n" + "\n\n---\n\n".join(relevant_texts)
    user_prompt += _get_cert_context(certification_type)

    result = call_llm_json(INCOME_SYSTEM_PROMPT, user_prompt, settings)
    result = validation.validate_income(result)

    si = result.get("sourceIncome", {})
    logger.info(
        "Extracted %d pay stubs, %d verification income records",
        len(si.get("payStub", [])),
        len(si.get("verificationIncome", [])),
    )
    return IncomeExtraction.model_validate(result)


def extract_assets(
    groups: list[DocumentGroup],
    settings: Settings,
    certification_type: str | None = None,
) -> AssetExtraction:
    """Extract asset data from pre-routed asset document groups."""
    relevant_texts = _build_texts(groups)
    if not relevant_texts:
        logger.info("No asset documents found")
        return AssetExtraction()

    user_prompt = "Extract asset data from these documents:\n\n" + "\n\n---\n\n".join(relevant_texts)
    user_prompt += _get_cert_context(certification_type)

    result = call_llm_json(ASSET_SYSTEM_PROMPT, user_prompt, settings)
    result = validation.validate_assets(result)

    logger.info(
        "Extracted %d asset records",
        len(result.get("assetInformation", [])),
    )
    return AssetExtraction.model_validate(result)


def extract_document_inventory_financial(
    groups: list[DocumentGroup],
    settings: Settings,
) -> DocumentInventory:
    """Catalog all financial/compliance documents (Section 23)."""
    # Send all non-HUD groups
    hud_types = {
        "HUD 9887", "HUD 9887-A", "HUD 92006",
        "HUD Race and Ethnic Data Form",
        "Acknowledgement of Receipt of HUD Forms",
        "Authorization to Release Information",
        "Tenant Release and Consent Form",
        "EIV Summary Report", "EIV Income Report",
    }

    relevant_texts = []
    for g in groups:
        if g.document_type in hud_types:
            continue
        relevant_texts.append(
            f"[Pages: {g.page_range}, Classified as: {g.document_type}, "
            f"Category: {g.category}, Person: {g.person_name or 'Unknown'}]\n"
            f"{g.combined_text}"
        )

    if not relevant_texts:
        return DocumentInventory()

    user_prompt = (
        "Catalog these financial and compliance documents:\n\n"
        + "\n\n---\n\n".join(relevant_texts)
    )

    result = call_llm_json(INVENTORY_FINANCIAL_SYSTEM_PROMPT, user_prompt, settings)
    result = validation.validate_document_inventory(result)

    logger.info(
        "Financial inventory: %d documents cataloged",
        len(result.get("documents", [])),
    )
    return DocumentInventory.model_validate(result)


def extract_document_inventory_hud(
    groups: list[DocumentGroup],
    settings: Settings,
) -> DocumentInventory:
    """Catalog HUD compliance forms only (Section 24)."""
    hud_types = {
        "HUD 9887", "HUD 9887-A", "HUD 92006",
        "HUD Race and Ethnic Data Form",
        "Acknowledgement of Receipt of HUD Forms",
        "Authorization to Release Information",
        "Tenant Release and Consent Form",
        "EIV Summary Report", "EIV Income Report",
    }

    relevant_texts = []
    for g in groups:
        if g.document_type in hud_types or g.category == "compliance":
            relevant_texts.append(
                f"[Pages: {g.page_range}, Classified as: {g.document_type}, "
                f"Person: {g.person_name or 'Unknown'}]\n{g.combined_text}"
            )

    if not relevant_texts:
        return DocumentInventory()

    user_prompt = (
        "Catalog HUD compliance forms from these documents:\n\n"
        + "\n\n---\n\n".join(relevant_texts)
    )

    result = call_llm_json(INVENTORY_HUD_SYSTEM_PROMPT, user_prompt, settings)
    result = validation.validate_document_inventory(result)

    logger.info(
        "HUD inventory: %d documents cataloged",
        len(result.get("documents", [])),
    )
    return DocumentInventory.model_validate(result)
