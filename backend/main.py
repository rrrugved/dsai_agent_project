from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.schemas import AgentResponse
from backend.service import run_agent_request, stream_agent_request

app = FastAPI(title="DSAI Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=AgentResponse)
async def chat(
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
):
    active_session_id = session_id or str(uuid.uuid4())
    result = run_agent_request(
        query=query,
        session_id=active_session_id,
        uploaded_files=files,
    )
    return result


@app.post("/chat/stream")
async def chat_stream(
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
):
    active_session_id = session_id or str(uuid.uuid4())

    def _event_stream():
        for event in stream_agent_request(
            query=query,
            session_id=active_session_id,
            uploaded_files=files,
        ):
            yield event

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)