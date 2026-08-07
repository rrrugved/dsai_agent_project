import os
import base64
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from an image file using Gemini Vision AI.
    Use this tool when a user provides an image file (png, jpg, jpeg) or a scanned document.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    try:
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        ext = file_path.split(".")[-1].lower()
        mime_type = f"image/{'jpeg' if ext in ['jpg', 'jpeg'] else ext}"
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe all visible text from this image exactly as it appears. Maintain the formatting. Return only the extracted text."},
                {"type": "image_url", "image_url": f"data:{mime_type};base64,{base64_image}"}
            ]
        )
        
        response = llm.invoke([message])
        
        extracted_text = response.content.strip()
        if not extracted_text:
            return "Vision AI completed but no text could be extracted from the image."
            
        return extracted_text

    except Exception as e:
        return f"Error processing image with Vision AI: {str(e)}"

if __name__ == "__main__":
    print("Starting script...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is missing or empty in .env!")
    else:
        print(f"GOOGLE_API_KEY found (starts with: {api_key[:6]}...)")

    sample_image_path = r"C:\Users\HP\OneDrive\Desktop\DataSmithAi ass\dsai_agent_project\ocr-test.png"
    print(f"Checking file at: {sample_image_path}")
    
    if os.path.exists(sample_image_path):
        print("File exists! Sending request to Gemini API...")
        result = extract_text_from_image.invoke({"file_path": sample_image_path})
        print("Done! Result below:\n")
        print(result)
    else:
        print(f"[ERROR] Image file not found at {sample_image_path}")