"""Central text sanitizer — cleans OCR output for all downstream extractors.

Applied once during page grouping so every extractor (parsers, LLM, Vision)
works on consistent clean text. The raw HTML is preserved in a separate field
for parsers that need table structure.
"""

import re


def sanitize_for_extraction(text: str) -> str:
    """Clean OCR text for LLM and regex extraction.

    Removes:
    - OCR escape artifacts: \\( \\) \\[ \\]
    - HTML entities: &amp; &lt; &gt; &#x27; etc.
    - Grounding/detection tags: <|ref|>, <|det|>
    - Excessive whitespace

    Preserves:
    - HTML table structure (for table_utils parsers)
    - Line breaks (important for section detection)
    """
    if not text:
        return ""

    # 1. Strip grounding/detection tags (OCR model artifacts)
    clean = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", text, flags=re.DOTALL)
    clean = re.sub(r"<\|det\|>.*?<\|/det\|>", "", clean, flags=re.DOTALL)

    # 2. Decode HTML entities
    clean = clean.replace("&amp;amp;", "&")
    clean = clean.replace("&amp;", "&")
    clean = clean.replace("&lt;", "<")
    clean = clean.replace("&gt;", ">")
    clean = clean.replace("&#x27;", "'")
    clean = clean.replace("&quot;", '"')
    clean = clean.replace("&#39;", "'")

    # 3. Fix OCR escape artifacts: \\( \\) → remove
    clean = re.sub(r"\\+[()]", "", clean)

    # 4. Collapse runs of whitespace (but preserve newlines)
    clean = re.sub(r"[ \t]+", " ", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)

    return clean.strip()


def strip_html(text: str) -> str:
    """Remove all HTML tags, returning plain text.

    Use this when you need pure text (e.g., for keyword matching, LLM prompts).
    For table parsing, use the raw text with sanitize_for_extraction() instead.
    """
    clean = sanitize_for_extraction(text)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def clean_extracted_value(value: str | None) -> str | None:
    """Clean a single extracted field value — remove HTML fragments and artifacts."""
    if value is None:
        return None

    val = str(value).strip()
    if not val:
        return None

    # Remove HTML tags
    val = re.sub(r"<[^>]+>", "", val)
    # Remove HTML entities
    val = val.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    val = val.replace("&#x27;", "'").replace("&quot;", '"')
    # Remove OCR escapes
    val = re.sub(r"\\+[()]", "", val)
    # Collapse whitespace
    val = re.sub(r"\s+", " ", val).strip()

    # If result is empty or just punctuation, return None
    if not val or len(val) < 2 or all(c in ".,;:!?- " for c in val):
        return None

    return val
