"""Income calculation engine — computes annual income using four methods (Section 9)."""

import logging
from datetime import date, datetime

from app.schemas.extraction import (
    IncomeCalculationResult,
    PayStubEntry,
    VerificationIncomeEntry,
)
from app.services.hours_resolver import resolve_hours_range

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frequency multipliers
# ---------------------------------------------------------------------------

FREQUENCY_MULTIPLIERS: dict[str, int] = {
    "weekly": 52,
    "bi-weekly": 26,
    "semi-monthly": 24,
    "monthly": 12,
    "quarterly": 4,
    "annually": 1,
}


def get_frequency_multiplier(frequency: str | None) -> int | None:
    """Return the annual multiplier for a pay frequency."""
    if not frequency:
        return None
    return FREQUENCY_MULTIPLIERS.get(frequency.strip().lower())


# ---------------------------------------------------------------------------
# Individual calculation methods
# ---------------------------------------------------------------------------

def calculate_self_declared(
    amount: str | None,
    frequency: str | None = None,
) -> str | None:
    """Self-declared income annualized by frequency.

    The LLM often extracts fixed-benefit amounts (SSA, pension, child support)
    into selfDeclaredAmount with frequencyOfPay="monthly". A monthly $1,414
    must become $16,968 annual, not $1,414. When frequency is missing or
    "annually", the amount is returned as-is.
    """
    if not amount:
        return None
    try:
        val = float(amount)
    except ValueError:
        return None
    mult = get_frequency_multiplier(frequency) if frequency else None
    if mult and mult > 1:
        val = val * mult
    return f"{val:.2f}"


def calculate_voi_based(
    rate_of_pay: str | None,
    frequency_of_pay: str | None,
    hours_per_pay_period: str | None,
    overtime_rate: str | None = None,
    overtime_frequency: str | None = None,
    funding_program: str | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """VOI-based annual income: rate × hours × frequency_multiplier + overtime.

    Returns:
        (annual_income, details_string, list_of_findings)
    """
    findings: list[str] = []

    if not rate_of_pay or not frequency_of_pay:
        return None, None, findings

    try:
        rate = float(rate_of_pay)
    except ValueError:
        return None, None, findings

    multiplier = get_frequency_multiplier(frequency_of_pay)
    if multiplier is None:
        return None, f"Unknown frequency: {frequency_of_pay}", findings

    # Resolve hours (may be a range)
    hours = None
    if hours_per_pay_period:
        hours, hours_finding = resolve_hours_range(hours_per_pay_period, funding_program)
        if hours_finding:
            findings.append(hours_finding)

    # Calculate base income.
    # hours_per_pay_period is hours in ONE pay period (Work Number's
    # "Avg Hours Worked/Pay Period" field — e.g. 80 for a bi-weekly worker
    # averaging 40 hrs/week). Annualize via the period multiplier, not a
    # fixed ×52 which would only be correct for weekly frequencies.
    if hours is not None:
        annual = rate * hours * multiplier
        details = f"{rate} × {hours} hrs/pp × {multiplier} pp/yr = {annual:.2f}"
    else:
        # Periodic rate: rate × multiplier
        annual = rate * multiplier
        details = f"{rate} × {multiplier} periods = {annual:.2f}"

    # Add overtime
    overtime_annual = 0.0
    if overtime_rate:
        try:
            ot_rate = float(overtime_rate)
            ot_multiplier = get_frequency_multiplier(overtime_frequency) or multiplier
            overtime_annual = ot_rate * ot_multiplier
            details += f" + OT {ot_rate} × {ot_multiplier} = {overtime_annual:.2f}"
        except ValueError:
            pass

    total = annual + overtime_annual
    return f"{total:.2f}", details, findings


def calculate_ytd_based(
    ytd_amount: str | None,
    ytd_start_date: str | None,
    ytd_end_date: str | None,
) -> tuple[str | None, str | None]:
    """YTD-based annual income: ytd_amount / days_elapsed × 365.

    Returns:
        (annual_income, details_string)
    """
    if not ytd_amount:
        return None, None

    try:
        ytd = float(ytd_amount)
    except ValueError:
        return None, None

    start = _parse_date(ytd_start_date)
    end = _parse_date(ytd_end_date)

    if not start or not end:
        return None, "YTD dates missing — cannot annualize"

    days = (end - start).days
    if days <= 0:
        return None, f"Invalid YTD period: {ytd_start_date} to {ytd_end_date}"

    annual = ytd / days * 365
    details = f"{ytd:.2f} / {days} days × 365 = {annual:.2f}"
    return f"{annual:.2f}", details


def calculate_paystub_based(
    paystubs: list[PayStubEntry],
    pay_interval: str | None = None,
) -> tuple[str | None, str | None]:
    """Pay-stub-based annual income: average gross × frequency multiplier.

    Args:
        paystubs: list of PayStubEntry for one income source
        pay_interval: override frequency (if not on individual stubs)

    Returns:
        (annual_income, details_string)
    """
    if not paystubs:
        return None, None

    amounts = []
    freq = pay_interval
    for ps in paystubs:
        if ps.grossPay:
            try:
                amounts.append(float(ps.grossPay))
            except ValueError:
                continue
        if not freq and ps.payInterval:
            freq = ps.payInterval

    if not amounts:
        return None, None

    avg = sum(amounts) / len(amounts)
    multiplier = get_frequency_multiplier(freq)
    if multiplier is None:
        return None, f"Unknown pay interval: {freq}"

    annual = avg * multiplier
    details = f"avg({len(amounts)} stubs) = {avg:.2f} × {multiplier} = {annual:.2f}"
    return f"{annual:.2f}", details


# ---------------------------------------------------------------------------
# Main orchestrator — compute all applicable methods for one income source
# ---------------------------------------------------------------------------

def calculate_all_methods(
    vi_entry: VerificationIncomeEntry | None,
    matching_paystubs: list[PayStubEntry],
    funding_program: str | None = None,
) -> list[IncomeCalculationResult]:
    """Run all applicable income calculation methods for one source.

    Args:
        vi_entry: verification income entry (may be None if only paystubs)
        matching_paystubs: paystubs for this income source
        funding_program: for hours range resolution

    Returns:
        list of IncomeCalculationResult, one per method that produced a result
    """
    results: list[IncomeCalculationResult] = []
    member_name = None
    source_name = None

    if vi_entry:
        member_name = vi_entry.memberName
        source_name = vi_entry.sourceName

        # Classify income type for calculation routing
        income_type = (vi_entry.incomeType or "").lower()
        calc_mode = _classify_income_mode(income_type)

        # Method 1: Self-declared — routed by calc_mode.
        # For fixed_monthly types (SSA, pension, child support) the
        # selfDeclaredAmount is typically the ANNUAL total from a benefit
        # letter or TIC column, not a monthly figure. frequencyOfPay
        # describes when the person is PAID, not the amount's time basis.
        # Blindly ×12 here double-annualizes (e.g. $16,976/yr × 12 = $203,716).
        sd_amount = vi_entry.selfDeclaredAmount
        if sd_amount:
            try:
                sd_val = float(sd_amount)
                if calc_mode in ("fixed_monthly", "annual_net"):
                    # Already annual — use as-is
                    sd_str = f"{sd_val:.2f}"
                    sd_details = f"Self-declared annual: {sd_str}"
                else:
                    # Employment — annualize by frequencyOfPay
                    sd_str = calculate_self_declared(
                        sd_amount, vi_entry.frequencyOfPay,
                    )
                    sd_details = f"Self-declared amount: {sd_str}"
                if sd_str:
                    results.append(IncomeCalculationResult(
                        memberName=member_name,
                        sourceName=source_name,
                        method="self-declared",
                        annualIncome=sd_str,
                        details=sd_details,
                    ))
            except ValueError:
                pass

        # Method 2: VOI-based — route by income mode
        if calc_mode == "fixed_monthly" and vi_entry.rateOfPay:
            # Fixed monthly income (SSA, TANF, pension, child support)
            # rate = monthly amount, annual = rate × 12
            try:
                monthly = float(vi_entry.rateOfPay)
                annual = monthly * 12
                results.append(IncomeCalculationResult(
                    memberName=member_name,
                    sourceName=source_name,
                    method="voi-based",
                    annualIncome=f"{annual:.2f}",
                    details=f"Fixed monthly: {monthly:.2f} × 12 = {annual:.2f}",
                ))
            except ValueError:
                pass

        elif calc_mode == "annual_net":
            # Self-employment / business income
            # Priority: selfDeclaredAmount (TIC column A) > rateOfPay
            # The TIC already has the correct annual figure. The affidavit's
            # "net income" field is also annual. LLM often mislabels freq as
            # "monthly" causing ×12 inflation.
            try:
                # Use selfDeclaredAmount first — it's the TIC annual total
                sd = vi_entry.selfDeclaredAmount
                rate = vi_entry.rateOfPay
                if sd:
                    annual = float(sd)
                    details = f"Business self-declared annual: {annual:.2f} (from TIC/affidavit)"
                elif rate:
                    amount = float(rate)
                    freq = (vi_entry.frequencyOfPay or "").lower()
                    # For business income, assume amount is annual unless
                    # freq is clearly periodic (weekly/bi-weekly)
                    if freq in ("weekly", "bi-weekly"):
                        mult = get_frequency_multiplier(freq) or 1
                        annual = amount * mult
                        details = f"Business periodic: {amount:.2f} × {mult} = {annual:.2f}"
                    else:
                        # Monthly or annual or unspecified — treat as annual net
                        # (self-employment affidavits report annual net from Schedule C)
                        annual = amount
                        details = f"Business/self-employment annual net: {annual:.2f}"
                else:
                    annual = None

                if annual is not None:
                    results.append(IncomeCalculationResult(
                        memberName=member_name,
                        sourceName=source_name,
                        method="voi-based",
                        annualIncome=f"{annual:.2f}",
                        details=details,
                    ))
            except ValueError:
                pass

        elif calc_mode == "employment" and (vi_entry.rateOfPay or vi_entry.frequencyOfPay):
            # Employment income — rate × hours × 52 (hourly) or rate × multiplier (periodic)
            voi_annual, voi_details, voi_findings = calculate_voi_based(
                vi_entry.rateOfPay,
                vi_entry.frequencyOfPay,
                vi_entry.hoursPerPayPeriod,
                vi_entry.overtimeRate,
                vi_entry.overtimeFrequency,
                funding_program,
            )
            if voi_annual:
                results.append(IncomeCalculationResult(
                    memberName=member_name,
                    sourceName=source_name,
                    method="voi-based",
                    annualIncome=voi_annual,
                    details=voi_details,
                ))

        elif vi_entry.rateOfPay and vi_entry.frequencyOfPay:
            # Unknown type but has rate+freq — use frequency multiplier only (no hours)
            try:
                rate = float(vi_entry.rateOfPay)
                mult = get_frequency_multiplier(vi_entry.frequencyOfPay)
                if mult:
                    annual = rate * mult
                    results.append(IncomeCalculationResult(
                        memberName=member_name,
                        sourceName=source_name,
                        method="voi-based",
                        annualIncome=f"{annual:.2f}",
                        details=f"{rate:.2f} × {mult} periods = {annual:.2f}",
                    ))
            except ValueError:
                pass

        # Method 3: YTD-based (not for fixed or annual-net income)
        if calc_mode == "employment":
            ytd_annual, ytd_details = calculate_ytd_based(
                vi_entry.ytdAmount,
                vi_entry.ytdStartDate,
                vi_entry.ytdEndDate,
            )
            if ytd_annual:
                results.append(IncomeCalculationResult(
                    memberName=member_name,
                    sourceName=source_name,
                    method="ytd-based",
                    annualIncome=ytd_annual,
                    details=ytd_details,
                ))
    elif matching_paystubs:
        member_name = matching_paystubs[0].memberName
        source_name = matching_paystubs[0].sourceName

    # Method 4: Pay-stub-based
    if matching_paystubs:
        ps_annual, ps_details = calculate_paystub_based(matching_paystubs)
        if ps_annual:
            results.append(IncomeCalculationResult(
                memberName=member_name or (matching_paystubs[0].memberName if matching_paystubs else None),
                sourceName=source_name or (matching_paystubs[0].sourceName if matching_paystubs else None),
                method="paystub-based",
                annualIncome=ps_annual,
                details=ps_details,
            ))

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_income_mode(income_type: str) -> str:
    """Classify income type into a calculation mode.

    Returns:
        "fixed_monthly" — SSA, TANF, pension, child support: rate × 12
        "annual_net"    — self-employment, business: annual net from Schedule C
        "employment"    — wages: rate × hours × 52 or rate × frequency
    """
    if income_type in (
        "social security", "supplemental security income",
        "social security disability", "pension", "temporary assistance",
        "child support", "alimony", "ssi", "ssdi",
    ):
        return "fixed_monthly"

    if income_type in (
        "self-employment", "business", "business income",
    ):
        return "annual_net"

    # Default: employment (hourly/periodic wage)
    return "employment"


def _parse_date(value: str | None) -> date | None:
    """Parse a YYYY-MM-DD date string."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def match_paystubs_to_sources(
    paystubs: list[PayStubEntry],
    vi_entries: list[VerificationIncomeEntry],
) -> dict[int, list[PayStubEntry]]:
    """Match paystubs to verification income entries by source/member name.

    Returns:
        dict mapping vi_entry index → list of matching paystubs
    """
    matched: dict[int, list[PayStubEntry]] = {}
    unmatched: list[PayStubEntry] = list(paystubs)

    for i, vi in enumerate(vi_entries):
        matched[i] = []
        vi_source = (vi.sourceName or "").lower().strip()
        vi_member = (vi.memberName or "").lower().strip()

        if not vi_source and not vi_member:
            continue

        still_unmatched = []
        for ps in unmatched:
            ps_source = (ps.sourceName or "").lower().strip()
            ps_member = (ps.memberName or "").lower().strip()

            # Match by source name (fuzzy: one contains the other)
            source_match = False
            if vi_source and ps_source:
                source_match = (
                    vi_source in ps_source
                    or ps_source in vi_source
                    or _token_overlap(vi_source, ps_source) >= 0.5
                )

            # Match by member name
            member_match = False
            if vi_member and ps_member:
                member_match = vi_member == ps_member or _token_overlap(vi_member, ps_member) >= 0.5

            if source_match or (member_match and not vi_source):
                matched[i].append(ps)
            else:
                still_unmatched.append(ps)

        unmatched = still_unmatched

    return matched


def _token_overlap(a: str, b: str) -> float:
    """Compute token overlap ratio between two strings."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b)
    return overlap / min(len(tokens_a), len(tokens_b))
