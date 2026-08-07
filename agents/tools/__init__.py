"""files for tools that the agent can use to perform specific tasks and make all the files to text compatible with langchain_core tools"""
from .ocr_tool import extract_text_from_image
from .audio_tool import transcribe_audio
from .web_tool import fetch_webpage_content
from .yt_tool import fetch_youtube_transcript
from .pdf_parser import parse_pdf

__all__ = [
    "extract_text_from_image",
    "transcribe_audio",
    "fetch_webpage_content",
    "fetch_youtube_transcript",
    "parse_pdf"
]