# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install critical system dependencies (ffmpeg is REQUIRED for Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Install 'uv' package manager
RUN pip install --no-cache-dir uv

# 5. Copy your project files into the container
COPY . /app

# 6. Install Python dependencies system-wide inside the container using uv
RUN uv pip install --system \
    langchain-core \
    langchain-google-genai \
    openai-whisper \
    requests \
    beautifulsoup4 \
    python-dotenv

# 7. Expose the port your frontend/API will run on (e.g., FastAPI or Streamlit)
EXPOSE 8000

# 8. Define the default command to start your application
# (We will update this once you build your main server/graph file)
CMD ["python", "app.py"]