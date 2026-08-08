from __future__ import annotations

import io
import json
import os
import uuid
from typing import Any, Dict, List

import requests
import streamlit as st


FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Multi-Modal Agent", page_icon="💬", layout="centered")


def _new_chat() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.attached_files = []


def _init_state() -> None:
    if "session_id" not in st.session_state:
        _new_chat()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "attached_files" not in st.session_state:
        st.session_state.attached_files = []


def _mime_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".m4a":
        return "audio/mp4"
    return "application/octet-stream"


def _call_backend(query: str, files: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
    multipart_files = [
        ("files", (item["name"], io.BytesIO(item["bytes"]), item["mime_type"]))
        for item in files
    ]
    response = requests.post(
        f"{FASTAPI_URL.rstrip('/')}/chat",
        data={"query": query, "session_id": session_id},
        files=multipart_files or None,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _call_backend_stream(query: str, files: List[Dict[str, Any]], session_id: str):
    multipart_files = [
        ("files", (item["name"], io.BytesIO(item["bytes"]), item["mime_type"]))
        for item in files
    ]
    response = requests.post(
        f"{FASTAPI_URL.rstrip('/')}/chat/stream",
        data={"query": query, "session_id": session_id},
        files=multipart_files or None,
        stream=True,
        timeout=600,
    )
    response.raise_for_status()

    accumulated_text = ""
    final_payload: Dict[str, Any] = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        event = json.loads(raw_line)
        event_type = event.get("type")
        if event_type == "status":
            yield {"kind": "status", "text": event.get("text", "Agent is thinking...")}
        elif event_type == "token":
            accumulated_text += event.get("text", "")
            yield {"kind": "token", "text": accumulated_text}
        elif event_type == "result":
            final_payload = event.get("payload", {})
        elif event_type == "error":
            raise requests.RequestException(event.get("text", "Streaming request failed"))
    if not final_payload:
        final_payload = {
            "final_output": accumulated_text,
            "plan_trace": [],
            "extracted_text_map": {},
            "extracted_text": "",
        }
    yield {"kind": "final", "payload": final_payload}


def _source_label(source_name: str) -> str:
    lowered = source_name.lower()
    if lowered.startswith("you tube:") or lowered.startswith("youtube:"):
        return "YouTube Transcript"
    if lowered.startswith("web:"):
        return "Web Page"
    if lowered.startswith("vector database search results"):
        return "Retrieved Context"
    if ".pdf" in lowered:
        return "PDF"
    if lowered.endswith((".png", ".jpg", ".jpeg")):
        return "Image OCR"
    if lowered.endswith((".mp3", ".wav", ".m4a")):
        return "Audio Transcript"
    return "Source"


def _render_extracted_blocks(extracted_map: Dict[str, str]) -> None:
    if not extracted_map:
        st.markdown("_No extracted text returned._")
        return

    for source_name, content in extracted_map.items():
        if not content:
            continue
        with st.container(border=True):
            st.markdown(f"**{_source_label(source_name)}**")
            st.caption(source_name)
            st.markdown(content)


_init_state()

with st.sidebar:
    if st.button("New Chat", use_container_width=True):
        _new_chat()
        st.rerun()

    st.divider()
    st.markdown("## Attachments")
    uploads = st.file_uploader(
        "Upload files",
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Attach Files", use_container_width=True):
            if uploads:
                for item in uploads:
                    st.session_state.attached_files.append(
                        {
                            "name": item.name,
                            "bytes": item.getvalue(),
                            "mime_type": _mime_type(item.name),
                        }
                    )
                st.success("Files attached.")
            else:
                st.warning("Choose at least one file first.")

    with col_b:
        if st.button("Clear Files", use_container_width=True):
            st.session_state.attached_files = []
            st.rerun()

    if st.session_state.attached_files:
        st.caption("Attached files")
        for item in st.session_state.attached_files:
            st.markdown(f"- {item['name']}")
    else:
        st.caption("No attached files yet.")

    st.divider()
    st.caption(f"Backend: `{FASTAPI_URL}`")

st.title("Multi-Modal Agent Terminal")
st.markdown(
    "Ask a question, attach a PDF, image, or audio file, and the agent will "
    "extract the content, retrieve context, and answer in text only."
)
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Plan trace", expanded=False):
                for step in message.get("plan_trace", []):
                    st.write(f"- {step}")
            with st.expander("Extracted text", expanded=False):
                _render_extracted_blocks(message.get("extracted_text_map", {}))

if prompt := st.chat_input("Ask about your files or combine multiple inputs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            placeholder.markdown("Agent is thinking...")
            payload = {}
            final_output = ""
            for event in _call_backend_stream(prompt, st.session_state.attached_files, st.session_state.session_id):
                if event["kind"] == "status":
                    placeholder.markdown(event["text"])
                elif event["kind"] == "token":
                    final_output = event["text"]
                    placeholder.markdown(final_output + "▌")
                elif event["kind"] == "final":
                    payload = event["payload"]
                    final_output = payload.get("final_output", final_output or "No response generated.")
                    placeholder.markdown(final_output)

            assistant_message = {
                "role": "assistant",
                "content": final_output,
                "plan_trace": payload.get("plan_trace", []),
                "extracted_text": payload.get("extracted_text", ""),
                "extracted_text_map": payload.get("extracted_text_map", {}),
            }
            st.session_state.messages.append(assistant_message)

            with st.expander("Plan trace", expanded=False):
                for step in assistant_message["plan_trace"]:
                    st.write(f"- {step}")
            with st.expander("Extracted text", expanded=False):
                _render_extracted_blocks(assistant_message.get("extracted_text_map", {}))

        except requests.RequestException as exc:
            error_message = f"Backend request failed: {exc}"
            placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message, "plan_trace": [], "extracted_text": "", "extracted_text_map": {}})
