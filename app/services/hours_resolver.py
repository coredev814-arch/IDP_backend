"""Hours range resolver — applies funding-program-specific rules (Section 10)."""

import re
import logging

logger = logging.getLogger(__name__)

# Funding program → hours range resolution strategy
_PROGRAM_RULES: dict[str, str] = {
    "lihtc": "highest",
    "tax credit": "highest",
    "low-income housing tax credit": "highest",
    "hud": "average",
    "section 8": "average",
    "section 202": "average",
    "section 236": "average",
    "hud shp": "average",
    "usda": "average",
    "rad": "average",
    "rental assistance demonstration": "average",
    "public housing": "lowest",
    # PRAC is an exception within HUD — uses average
    "prac": "average",
}

_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)"
)


def resolve_hours_range(
    hours_text: str | None,
    funding_program: str | None = None,
) -> tuple[float | None, str | None]:
    """Resolve an hours value (possibly a range) to a single number.

    Args:
        hours_text: Hours text, e.g. "40", "30-35", "30 to 35"
        funding_program: Funding program name for range resolution

    Returns:
        (resolved_hours, finding_or_none)
        - resolved_hours: single float value, or None if unparseable
        - finding: a compliance finding string if range rule was applied, or None
    """
    if not hours_text:
        return None, None

    hours_text = hours_text.strip()
    finding = None

    # Check for range pattern
    m = _RANGE_PATTERN.search(hours_text)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        if low > high:
            low, high = high, low

        rule = _get_rule(funding_program)

        if rule == "highest":
            resolved = high
            finding = (
                f"Hours range {low}-{high} resolved to {high} (highest) "
                f"per {funding_program or 'LIHTC'} program rules (Section 10)"
            )
        elif rule == "lowest":
            resolved = low
            finding = (
                f"Hours range {low}-{high} resolved to {low} (lowest) "
                f"per {funding_program or 'Public Housing'} program rules (Section 10)"
            )
        else:  # average
            resolved = (low + high) / 2
            if not funding_program:
                finding = (
                    f"Hours range {low}-{high} resolved to {resolved} (average) — "
                    f"funding program not specified, defaulting to average (Section 10)"
                )
            else:
                finding = (
                    f"Hours range {low}-{high} resolved to {resolved} (average) "
                    f"per {funding_program} program rules (Section 10)"
                )

        return resolved, finding

    # Single number
    try:
        return float(hours_text), None
    except ValueError:
        # Try extracting just digits
        digits = re.findall(r"\d+(?:\.\d+)?", hours_text)
        if digits:
            return float(digits[0]), None
        return None, None


def _get_rule(funding_program: str | None) -> str:
    """Look up the resolution rule for a funding program."""
    if not funding_program:
        return "average"

    key = funding_program.strip().lower()
    for program, rule in _PROGRAM_RULES.items():
        if program in key or key in program:
            return rule

    return "average"
