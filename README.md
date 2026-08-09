# DSAI Multi-Modal Agent

A FastAPI and Streamlit application that accepts text, PDFs, images, audio, and URLs. It extracts source content, chooses a task, runs the minimum tool sequence, retrieves only the current request's evidence, and returns a text-only response with the extraction and plan trace.

## Features

- Text conversation with a chat-style Streamlit UI.
- PDF extraction with PyMuPDF; scanned pages use Tesseract OCR and Gemini Vision fallback.
- Image OCR for PNG, JPG, and JPEG files. Gemini does not provide a calibrated OCR confidence value, so image OCR reports `unavailable` rather than inventing one.
- MP3, WAV, and M4A transcription through Groq Whisper, plus duration detection through `ffprobe` (WAV has a standard-library fallback).
- URL discovery anywhere in the prompt or extracted document text. YouTube links use the transcript API; ordinary web URLs are scraped as readable text.
- Summarization, sentiment analysis, code explanation, Q&A, and audio-summary output formats.
- Current-request-scoped Qdrant retrieval: a PDF/audio/image upload cannot retrieve stale chunks from an earlier unrelated upload.
- Extracted text and a readable tool/decision trace shown in the UI.

## Architecture

```text
Prompt + attachments
        |
        v
Intent classifier --> clarification question (when task is ambiguous)
        |
        v
PDF / image / audio / URL tools
        |
        v
Qdrant ingestion + source-scoped retrieval
        |
        v
Relevance check --> final synthesizer --> API/UI response
```

The diagram assets are also available as `graph_architecture.png` and `graph_architecture2.png`.

## Retrieval design

All chunks are stored in the `agent_documents` Qdrant collection. Each source is assigned a content-derived `source_signature`, saved in chunk metadata, and indexed as a Qdrant `keyword` payload field.

For a request containing files, retrieval filters on only that request's signatures. This prevents a generic query such as “summarize this audio” from returning an older PDF or a failed YouTube transcript. Failed extraction/fetch responses are not ingested.

This project deliberately does **not** implement persistent document sessions. To ask about a file again in a later request, attach it again. Multi-file reasoning works when the relevant files are attached together in the same request.

## Prerequisites

- Python 3.11+
- `uv`
- A Google API key for Gemini chat, vision, and embeddings
- A Groq API key for audio transcription
- Qdrant Cloud credentials, or omit both Qdrant variables to use local Qdrant storage
- Tesseract and `ffmpeg`/`ffprobe` for local OCR and audio duration support (included in the provided Docker images)

## Configuration

Create `.env` in the project root:

```env
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Optional: leave both unset for local persistent Qdrant.
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...

# Optional, only when tesseract is not on PATH.
TESSERACT_CMD=/usr/bin/tesseract

# Streamlit uses this to call FastAPI.
BACKEND_URL=http://127.0.0.1:8000

# Maximum direct/explicitly requested URL fetches per request (default: 2).
MAX_EXTERNAL_URLS_PER_REQUEST=2
```

On startup the application creates the `metadata.source_signature` Qdrant payload index if it does not exist. This index is required by Qdrant Cloud for the filtered retrieval query.

To prevent a research paper's references from exhausting web/YouTube requests, URLs embedded in an uploaded document are fetched only when the prompt explicitly asks to use a link, URL, website, article, or video. Direct URLs typed in the prompt are eligible automatically. In either case, `MAX_EXTERNAL_URLS_PER_REQUEST` caps external fetches (default `2`), and skipped URLs appear in the plan trace.

## Run locally

Install the locked dependencies:

```bash
uv sync
```

Start the backend:

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

In a second terminal, start the UI:

```bash
uv run streamlit run frontend/app.py
```

Open the Streamlit URL printed in the terminal, usually `http://localhost:8501`.

## Docker

The repository includes separate images for the backend and frontend, plus a combined `Dockerfile` that runs FastAPI internally and exposes Streamlit on port `10000`.

```bash
docker build -t dsai-agent .
docker run --env-file .env -p 10000:10000 dsai-agent
```

For a public deployment, set the same environment variables in the deployment provider. The app binds to `0.0.0.0`; use the provider's assigned public URL as the deliverable.

## API

`POST /chat` accepts multipart form data:

- `query`: required text prompt
- `session_id`: optional UI conversation identifier
- `files`: zero or more PDF, PNG, JPG/JPEG, MP3, WAV, or M4A uploads

`POST /chat/stream` returns newline-delimited JSON status and final-result events. The current implementation streams status events and the final result; it does not token-stream the model response.

`GET /health` returns the backend health status.

## Assignment test coverage

| Assignment case | Expected behavior |
| --- | --- |
| Audio transcription + summary | Whisper transcript, duration, 1-line summary, 3 highlights, and a five-sentence summary. |
| PDF + question | Native text extraction or OCR fallback, then a source-scoped answer. |
| Image with code | OCR text is provided to the code-explanation task, including bug and complexity guidance. |
| PDF containing YouTube URL | PDF parsing discovers the URL and attempts transcript retrieval without another user prompt. If YouTube blocks access, the response records a clear fallback rather than fabricating a transcript. |
| PDF + audio comparison | Both sources are extracted and retrieved in the same request for a unified comparison. |

Run the integration suite when credentials and sample assets are available:

```bash
uv run pytest
```

The route-level tests are intentionally skipped when required external credentials are absent.

## Known external limitations

- YouTube may block transcript requests by IP or rate limit them. The app returns the API fallback message and continues with any remaining usable sources.
- OCR quality depends on scan resolution, language, and the installed Tesseract model.
- Qdrant, Gemini, Groq, and web/YouTube access are external services; their availability and quotas affect those portions of a request.
