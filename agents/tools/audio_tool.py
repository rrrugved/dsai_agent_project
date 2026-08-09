import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from groq import Groq

# Dynamically locate the project root directory containing .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio files (mp3, wav, m4a) into text using the Groq Whisper API.
    Use this tool when the user provides an audio file.
    """
    if not os.path.exists(file_path):
        return f"Error: Audio file not found at {file_path}"
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return f"Error: GROQ_API_KEY environment variable is not set. Looking in: {PROJECT_ROOT / '.env'}"
        
    try:
        client = Groq(api_key=api_key)
        
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(file_path), file.read()),
                model="whisper-large-v3-turbo",
            )
            
        transcript = transcription.text.strip()
        
        if not transcript:
            return "Transcription completed but no speech was detected."
            
        return transcript
    except Exception as e:
        return f"Error transcribing audio file: {str(e)}"

if __name__ == "__main__":
    sample_audio_path = r"C:\Users\HP\OneDrive\Desktop\DataSmithAi ass\dsai_agent_project\Record (online-voice-recorder.com).mp3"
    print("Testing Groq Whisper Transcription...\n")
    print(transcribe_audio.invoke({"file_path": sample_audio_path}))