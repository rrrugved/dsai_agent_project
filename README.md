# DSAI Agent Project

Multi-modal agent for text, PDF, image, audio, and YouTube-link workflows.

## What it does

- Parses PDFs and falls back to OCR for scanned pages.
- Extracts text from images.
- Transcribes audio.
- Detects YouTube links inside uploaded content and fetches transcripts.
- Stores chunked context in Qdrant and retrieves relevant chunks for synthesis.
- Produces text-only answers with a visible plan trace.

## Current scope

The project now includes:

- a FastAPI backend
- a Streamlit frontend
- route-level integration tests

## Setup

1. Create and activate the project environment.
2. Install dependencies with `uv sync`.
3. Add a `.env` file with the required API keys and service URLs.

## Environment variables

Required:

- `GOOGLE_API_KEY` - needed for Gemini text, vision, and embeddings calls.
- `QDRANT_URL` - Qdrant Cloud URL or local endpoint.
- `QDRANT_API_KEY` - Qdrant Cloud API key.

Optional:

- `FASTAPI_URL` - Streamlit backend URL, defaults to `http://127.0.0.1:8000`.
- `TESSERACT_CMD` - full path to the Tesseract executable if it is not already on `PATH`.

## Run the FastAPI backend

```bash
uv run uvicorn backend.main:app --reload
```

## Run the Streamlit frontend

```bash
uv run streamlit run frontend/app.py
```

## Run the terminal app

```bash
uv run python main.py
```

## Tests

Run the route-level tests with:

```bash
uv run pytest
```

The integration tests cover:

- PDF + YouTube link chaining
- Audio transcription + summary
- OCR fallback for scanned PDFs

The tests are skip-safe if the required external credentials are missing.

## Design notes

- Raw extracted text is used for ingestion only.
- Retrieved Qdrant chunks are used for final synthesis.
- Retrieval depth is chosen dynamically so small documents can surface more of their content without overloading the prompt.
- The agent prefers clarification over guessing when the input is ambiguous.

## What `schemas.py` is for

`backend/schemas.py` defines the Pydantic response model shared by the FastAPI backend and Streamlit frontend. It keeps the API response shape stable for the UI.

