import io
import os
import tempfile
from statistics import mean
from langchain_core.tools import tool
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from PIL import Image
import pytesseract


def _clean_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def _ocr_page_with_tesseract(page) -> tuple[str, float | None]:
    """Run OCR over a rendered page and return text plus average confidence."""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    confs = []

    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        word = str(text).strip()
        if not word:
            continue
        words.append(word)
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_val >= 0:
            confs.append(conf_val)

    return " ".join(words).strip(), (mean(confs) if confs else None)

def _ocr_page_with_vision(page) -> str:
    """Fallback OCR using the existing image transcription tool if local OCR fails."""
    from agents.tools.ocr_tool import extract_text_from_image

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path)

    try:
        return extract_text_from_image.invoke({"file_path": temp_path})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

@tool
def parse_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz)
    use this tool when the user provides a PDF file."""
    try:
        doc = fitz.open(file_path)
        page_sections = []
        confidences = []

        for page_number, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text("text"))
            if text:
                page_sections.append(f"[Page {page_number} | OCR Confidence: 100%]\n{text}")
                confidences.append(100.0)
                continue

            ocr_text, ocr_confidence = _ocr_page_with_tesseract(page)
            if not ocr_text:
                ocr_text = _clean_text(_ocr_page_with_vision(page))
                ocr_confidence = None

            if ocr_confidence is not None:
                confidences.append(ocr_confidence)
                conf_label = f"{ocr_confidence:.1f}%"
            else:
                conf_label = "unavailable"

            page_sections.append(f"[Page {page_number} | OCR Confidence: {conf_label}]\n{_clean_text(ocr_text)}")

        doc.close()

        final_text = "\n\n".join(section for section in page_sections if section.strip())
        if not final_text.strip():
            return "Fallback: PDF appears to be an image or scanned document. OCR processing required."

        overall_confidence = f"{mean(confidences):.1f}%" if confidences else "unavailable"
        return f"OCR Confidence: {overall_confidence}\n\n{final_text}"
        
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


if __name__ == "__main__":
    sample_pdf_path = r"C:\Users\HP\OneDrive\Desktop\DataSmithAi ass\Openclaw_Research_Report.pdf"
    print(f"Testing PDF Parser on {sample_pdf_path}...\n")
    print(parse_pdf(sample_pdf_path))
