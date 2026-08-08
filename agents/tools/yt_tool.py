import re
import multiprocessing as mp
from queue import Empty
from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from requests import Session


class _TimeoutSession(Session):
    """Requests session that prevents an unavailable YouTube request from stalling the graph."""

    def request(self, *args, **kwargs):
        kwargs["timeout"] = kwargs.get("timeout") or 3
        return super().request(*args, **kwargs)


def _download_transcript(video_id: str, result_queue) -> None:
    """Runs in a child process so an unresponsive network stack can be stopped."""
    try:
        ytt_api = YouTubeTranscriptApi(http_client=_TimeoutSession())
        transcript = ytt_api.fetch(video_id)
        result_queue.put(("ok", TextFormatter().format_transcript(transcript)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))

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
    
    result_queue = mp.Queue()
    process = mp.Process(target=_download_transcript, args=(video_id, result_queue))
    try:
        process.start()
        try:
            status, payload = result_queue.get(timeout=60)
        except Empty:
            return f"Fallback: Timed out while fetching transcript for video ID '{video_id}'."

        if status == "ok":
            return payload
        return f"Fallback: Could not fetch transcript for video ID '{video_id}'. Error: {payload}"
        
    except Exception as e:
        return f"Fallback: Could not fetch transcript for video ID '{video_id}'. Error: {str(e)}"
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        result_queue.close()

if __name__ == "__main__":
    sample_input = "Please hit this YT URL in the document and give me a summary: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print("Testing YouTube Tool...\n")
    print(fetch_youtube_transcript.invoke(sample_input))
