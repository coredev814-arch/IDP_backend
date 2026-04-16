from typing import Any, Optional, Union

from pydantic import BaseModel

from app.schemas.extraction import ExtractionResult


class PageDetail(BaseModel):
    page: int
    processed_image: str
    text_file: str
    text: str
    flag: Optional[str] = None
    flag_message: Optional[str] = None
    flag_details: list[Union[str, dict[str, Any]]] = []
    score: Optional[Union[float, dict]] = None


class FlaggedPage(BaseModel):
    page: int
    flag: str
    flag_message: Optional[str] = None
    score: Optional[float] = None


class ProcessPdfResponse(BaseModel):
    total_pages: int
    pages: list[PageDetail]
    summary: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
    flagged_pages: list[FlaggedPage] = []


class OcrResult(BaseModel):
    total_pages: int
    pages: list[PageDetail]
    summary: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
    flagged_pages: list[FlaggedPage] = []


class FullPipelineResponse(BaseModel):
    """Response from the full pipeline: OCR + Classification + Extraction."""
    ocr: OcrResult
    extraction: ExtractionResult
