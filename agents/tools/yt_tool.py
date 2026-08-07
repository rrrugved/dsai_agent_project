import re
from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

@tool
def fetch_youtube_transcript(text_input: str) -> str:
    """
    Extracts the transcript from a YouTube video URL found in the input text.
    Use this tool when a YouTube URL is present in a document or user query.
    """
    yt_pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(yt_pattern, text_input)
    
    if not match:
        return "No YouTube URL detected in the provided input."
        
    video_id = match.group(1)
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        
        formatter = TextFormatter()
        clean_text = formatter.format_transcript(transcript)
        
        return clean_text
        
    except Exception as e:
        return f"Fallback: Could not fetch transcript for video ID '{video_id}'. Error: {str(e)}"

if __name__ == "__main__":
    sample_input = "Please hit this YT URL in the document and give me a summary: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Testing YouTube Tool...\n")
    print(fetch_youtube_transcript(sample_input))