import os
import json
import subprocess
import wave
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_audio_duration_seconds(file_path: str) -> float | None:
    """Return audio duration using ffprobe, with WAV support as a stdlib fallback."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "json", file_path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        duration = float(json.loads(result.stdout)["format"]["duration"])
        return duration if duration >= 0 else None
    except (FileNotFoundError, subprocess.SubprocessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass

    if Path(file_path).suffix.lower() == ".wav":
        try:
            with wave.open(file_path, "rb") as audio:
                return audio.getnframes() / audio.getframerate()
        except (wave.Error, OSError, ZeroDivisionError):
            pass
    return None

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
