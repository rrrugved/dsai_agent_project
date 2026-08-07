import os
from langchain_core.tools import tool

@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribes audio files (mp3, wav, m4a) into text using the Whisper model.
    Use this tool when the user provides an audio file.
    """
    if not os.path.exists(file_path):
        return f"Error: Audio file not found at {file_path}"
        
    try:
        import whisper
        
        model = whisper.load_model("base")
        
        result = model.transcribe(file_path, fp16=False) 
        
        transcript = result.get("text", "").strip()
        
        if not transcript:
            return "Transcription completed but no speech was detected."
            
        return transcript
    except Exception as e:
        return f"Error transcribing audio file: {str(e)}"

if __name__ == "__main__":
    sample_audio_path = r"C:\Users\HP\OneDrive\Desktop\DataSmithAi ass\dsai_agent_project\sample-10s.mp3"
    print("Testing Whisper Transcription...\n")
    print(transcribe_audio.invoke({"file_path": sample_audio_path}))