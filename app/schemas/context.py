"""Pipeline context — carries request-scoped parameters through the pipeline."""

from typing import Optional

from pydantic import BaseModel


class PipelineContext(BaseModel):
    """Parameters that flow through the entire extraction pipeline."""
    funding_program: Optional[str] = None  # LIHTC, HUD, USDA, RAD, Public Housing
    certification_type: Optional[str] = None  # MI, AR, AR-SC, IR (from API or extracted)
