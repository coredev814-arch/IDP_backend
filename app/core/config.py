from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IDP_")

    app_name: str = "IDP - Intelligent Document Processing"
    app_version: str = "1.0.0"
    debug: bool = False

    # Storage
    output_dir: Path = Path("output")

    # PDF rendering
    # 200 DPI matches the native resolution of most scanned HUD/affordable
    # housing PDFs and provides ~1.3× supersample headroom for a clean
    # downsample to the 1280×1920 OCR target. Rendering higher just upsamples
    # the embedded scan — no real detail is recovered, and preprocessing
    # runtime scales with pixel count (~2.25× faster than 300 DPI).
    render_dpi: int = 200

    # Vision fallback for low-quality OCR pages.
    # When a page's OCR composite score falls below this threshold, the page
    # image is sent to Claude Vision for text extraction before classification.
    # This replaces the bad OCR text at the source so every downstream step
    # (classification, extraction, scoring) benefits from clean text.
    ocr_vision_threshold: float = 0.5

    # Image pre-processing
    # DeepSeek-OCR is optimized for 1280×1920 input. Smaller images force
    # the model to upscale internally which degrades accuracy on small text,
    # signatures, and handwritten fields. 1280×1920 matches the model's
    # native vision encoder resolution for best OCR quality.
    image_max_width: int = 1280
    image_max_height: int = 1920

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # OCR service
    ocr_service_url: str = ""
    ocr_fallback_url: str = ""
    ocr_prompt: str = "document"
    ocr_dpi: int = 200
    ocr_timeout: int = 600
    ocr_raw: bool = False
    ocr_retry: bool = True
    # OCR service can handle 4 concurrent requests. Parallelizing OCR
    # reduces whole-pipeline latency dramatically (~5x on 50+ page files).
    ocr_concurrency: int = 4
    # Image preprocessing runs in parallel worker threads because OpenCV
    # releases the Python GIL. Rendering stays sequential (PyMuPDF is not
    # thread-safe) but render(n+1) overlaps with preprocess(n).
    preprocess_concurrency: int = 4

    # LLM extraction (Claude API)
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    # Classification runs on Haiku by default — classification is a
    # constrained labeling task that Haiku handles well, and Haiku uses a
    # separate capacity pool so classification stays up when Sonnet 529s.
    llm_classify_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.0

    # Pipeline context (optional overrides)
    funding_program: str = ""  # LIHTC, HUD, USDA, RAD, Public Housing
    certification_type_override: str = ""  # MI, AR, AR-SC, IR

    # Upload validation
    allowed_content_types: list[str] = [
        "application/pdf",
        "application/octet-stream",
    ]
