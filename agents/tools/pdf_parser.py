import fitz
from langchain_core.tools import tool

@tool
def parse_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file using PyMuPDF (fitz)
    use this tool when the user provides a PDF file."""
    try:
        doc = fitz.open(file_path)
        extracted_text = []
        
        for page in doc:
            text = page.get_text()
            extracted_text.append(text)
            
        doc.close()
        final_text = "\n".join(extracted_text)
        
        if not final_text.strip():
            return "Fallback: PDF appears to be an image or scanned document. OCR processing required."
            
        return final_text
        
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


if __name__ == "__main__":
    sample_pdf_path = r"C:\Users\HP\OneDrive\Desktop\DataSmithAi ass\Openclaw_Research_Report.pdf"
    print(f"Testing PDF Parser on {sample_pdf_path}...\n")
    print(parse_pdf(sample_pdf_path))