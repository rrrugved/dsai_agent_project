import os
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def fetch_webpage_content(url: str) -> str:
    """
    Fetches and extracts clean text content from any standard website URL.
    Use this tool when a URL found inside a document or user query points to a webpage or article.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Error: Unable to reach webpage. HTTP Status Code: {response.status_code}"
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
            
        text = soup.get_text(separator="\n")
        
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        
        if len(clean_text) > 10000:
            clean_text = clean_text[:10000] + "\n...[Content Truncated]"
            
        return clean_text if clean_text else "Webpage loaded but no readable text found."
        
    except Exception as e:
        return f"Error fetching webpage content: {str(e)}"


if __name__ == "__main__":
    sample_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(fetch_webpage_content.invoke({"url": sample_url}))