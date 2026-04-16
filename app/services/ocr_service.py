import io
import logging
from pathlib import Path

import httpx
from PIL import Image

from app.core.config import Settings
from app.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)


def ocr_single_image(image_path: Path, settings: Settings) -> dict:                                                                                                                                          
    """Send a single processed image to the OCR service and return result with flags."""                                                                                                                     
    img = Image.open(image_path)                                                                                                                                                                             
                                                                                                                                                                                                            
    buf = io.BytesIO()                                                                                                                                                                                       
    img.save(buf, format="PNG")                                                                                                                                                                              
    buf.seek(0)                                                                                                                                                                                              

    try:                                                                                                                                                                                                     
        with httpx.Client(timeout=settings.ocr_timeout) as client:                                                                                                                                           
            response = client.post(                                                                                                                                                                          
                settings.ocr_service_url,                                                                                                                                                                    
                files={"file": (image_path.name, buf, "image/png")},                                                                                                                                         
                data={                                                                                                                                                                                     
                    "prompt": settings.ocr_prompt,
                    "dpi": str(settings.ocr_dpi),
                    "raw": bool(settings.ocr_raw),                                                                                                                                                           
                    "retry": bool(settings.ocr_retry),
                },                                                                                                                                                                                           
            )                                                                                                                                                                                              
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise ProcessingError(f"OCR service timed out for {image_path.name}") from exc                                                                                                                       
    except httpx.HTTPStatusError as exc:
        raise ProcessingError(                                                                                                                                                                               
            f"OCR service returned {exc.response.status_code} for {image_path.name}: "                                                                                                                     
            f"{exc.response.text}"                                                                                                                                                                           
        ) from exc
    except httpx.RequestError as exc:
        raise ProcessingError(f"OCR service unreachable: {exc}") from exc

    result = response.json()

    # When DeepSeek fails entirely, the OCR service sets needs_external_ocr.
    # Surface this as an ocr_failed flag so the classifier's existing
    # ocr_failed handling kicks in (pages are sent to the LLM with the flag
    # as metadata instead of being silently dropped).
    if result.get("needs_external_ocr"):
        flags = result.setdefault("flag_details", [])
        if isinstance(flags, list) and "ocr_failed" not in flags:
            flags.append("ocr_failed")

    return result


def ocr_job(job_id: str, settings: Settings) -> dict:
    """Run OCR on all processed page images for a given job.

    Returns a dict with job_id, total_pages, and per-page extracted text.
    """
    processed_dir = settings.output_dir / job_id / "processed"
    if not processed_dir.exists():
        raise ProcessingError(f"No processed images found for job '{job_id}'.")

    image_files = sorted(
        processed_dir.glob("page_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not image_files:
        raise ProcessingError(f"No page images found in job '{job_id}'.")

    logger.info("Starting OCR job=%s pages=%d", job_id, len(image_files))
    pages = []

    for image_path in image_files:
        page_num = int(image_path.stem.split("_")[1])
        logger.info("OCR job=%s page=%d", job_id, page_num)

        result = ocr_single_image(image_path, settings)

        pages.append({
            "page": page_num,
            "text": result.get("text"),
            "image": str(image_path),
            "flag": result.get("flag"),
            "flag_message": result.get("flag_message"),
            "flag_details": result.get("flag_details", []),
            "score": result.get("score"),
        })

    logger.info("OCR complete job=%s pages=%d", job_id, len(pages))

    summary = {"green": 0, "yellow": 0, "red": 0}
    flagged_pages =[]
    for p in pages:
        color = p.get("flag") or "yellow"
        summary[color] = summary.get(color, 0) + 1
        if color in ("yellow", "red"):
            flagged_pages.append({
                "page": p["page"],
                "flag": color,
                "flag_message": p.get("flag_message"),
                "score": p["score"]["composite"] if p.get("score") else None,
            })

    return {
        "job_id": job_id,
        "total_pages": len(pages),
        "pages": pages,
        "summary": summary,
        "flagged_pages": flagged_pages
    }
