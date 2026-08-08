import os
import tempfile
from pathlib import Path

import pytest
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont
from langchain_core.messages import HumanMessage

from agents.graph import graph


PDF_PATH = Path(__file__).resolve().parents[1] / "LLM_Guide_and_Video_Links-v2.pdf"
AUDIO_PATH = Path(__file__).resolve().parents[1] / "sample-10s.mp3"
REQUIRED_ENV_VARS = ("GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


def _create_scanned_pdf(temp_dir: Path) -> Path:
    image_path = temp_dir / "scanned_page.png"
    pdf_path = temp_dir / "scanned_sample.pdf"

    image = Image.new("RGB", (1400, 800), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (80, 120),
        "Scanned OCR fallback works.\nThis PDF contains no native text layer.",
        fill="black",
        font=font,
        spacing=18,
    )
    image.save(image_path)

    doc = fitz.open()
    page = doc.new_page(width=image.width, height=image.height)
    page.insert_image(page.rect, filename=str(image_path))
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.mark.skipif(
    any(not os.getenv(name) for name in REQUIRED_ENV_VARS),
    reason="Route-level integration test requires Google and Qdrant credentials.",
)
def test_route_level_pdf_and_youtube_flow():
    assert PDF_PATH.exists(), f"Expected PDF to exist at {PDF_PATH}"

    config = {"configurable": {"thread_id": "route-level-smoke-test"}}
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Hit the YouTube link in the attached PDF and give me a summary of it."
                    )
                )
            ],
            "file_paths": [str(PDF_PATH)],
        },
        config=config,
    )

    trace = result.get("plan_trace", [])
    final_text = result["messages"][-1].content

    assert any("parse_pdf" in step for step in trace)
    assert any("fetch_youtube_transcript" in step for step in trace)
    assert any("Retrieval depth selected dynamically" in step for step in trace)
    assert any("Synthesized final structured answer" in step for step in trace)
    assert result.get("retrieved_context")
    assert "Could you please clarify" not in final_text
    assert "1-line Summary" in final_text
    assert "Key Highlights" in final_text
    assert "Detailed Summary" in final_text


@pytest.mark.skipif(
    any(not os.getenv(name) for name in REQUIRED_ENV_VARS),
    reason="Route-level integration test requires Google and Qdrant credentials.",
)
def test_route_level_audio_summary_flow():
    assert AUDIO_PATH.exists(), f"Expected audio file to exist at {AUDIO_PATH}"

    config = {"configurable": {"thread_id": "route-level-audio-test"}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Summarize the attached audio.")],
            "file_paths": [str(AUDIO_PATH)],
        },
        config=config,
    )

    trace = result.get("plan_trace", [])
    final_text = result["messages"][-1].content

    assert any("transcribe_audio" in step for step in trace)
    assert any("Synthesized final structured answer" in step for step in trace)
    assert "Transcription" in final_text
    assert "1-line Summary" in final_text
    assert "Detailed Summary" in final_text


@pytest.mark.skipif(
    any(not os.getenv(name) for name in REQUIRED_ENV_VARS),
    reason="Route-level integration test requires Google and Qdrant credentials.",
)
def test_route_level_scanned_pdf_ocr_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        scanned_pdf = _create_scanned_pdf(tmp_dir)

        config = {"configurable": {"thread_id": "route-level-scanned-pdf-test"}}
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="What does the scanned PDF say?")],
                "file_paths": [str(scanned_pdf)],
            },
            config=config,
        )

    trace = result.get("plan_trace", [])
    final_text = result["messages"][-1].content

    assert any("parse_pdf" in step for step in trace)
    assert any("Synthesized final structured answer" in step for step in trace)
    assert "Scanned OCR fallback works" in final_text or "OCR fallback works" in final_text
