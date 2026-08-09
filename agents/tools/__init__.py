# agents/tools/__init__.py

from .ocr_tool import extract_text_from_image
from .pdf_parser import parse_pdf, _ocr_page_with_tesseract, _ocr_page_with_vision
from .audio_tool import transcribe_audio, get_audio_duration_seconds
from .web_tool import fetch_webpage_content
from .yt_tool import fetch_youtube_transcript

__all__ = [
    "extract_text_from_image",
    "parse_pdf",
    "transcribe_audio",
    "get_audio_duration_seconds",
    "fetch_webpage_content",
    "fetch_youtube_transcript",
    "ocr_page_with_tesseract",
    "ocr_page_with_vision",
]
