from __future__ import annotations

import json
import tempfile
import threading
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.messages import HumanMessage
from langsmith import traceable

from agents.graph import graph


def _write_upload_to_temp(uploaded_file, temp_dir: Path) -> Path:
    suffix = Path(uploaded_file.filename).suffix or ".bin"
    safe_name = Path(uploaded_file.filename).name
    # Two attachments can legitimately share a filename. Keep the display name
    # elsewhere, but give each temporary copy a unique filesystem name.
    temp_path = temp_dir / f"{uuid.uuid4().hex}_{safe_name}"
    if temp_path.suffix != suffix:
        temp_path = temp_path.with_suffix(suffix)
    temp_path.write_bytes(uploaded_file.file.read())
    return temp_path


def _format_extracted_text(extracted_text_map: Dict[str, str]) -> str:
    if not extracted_text_map:
        return ""
    sections: List[str] = []
    for source, content in extracted_text_map.items():
        if not content:
            continue
        sections.append(f"--- {source} ---\n{content}")
    return "\n\n".join(sections)


@traceable(name="dsai_agent_request", run_type="chain")
def _execute_graph_request(
    query: str,
    session_id: str,
    uploaded_files: Optional[List[object]] = None,
) -> Dict[str, object]:
    temp_paths: List[Path] = []

    try:
        with tempfile.TemporaryDirectory(prefix="dsai_agent_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            file_paths: List[str] = []

            for uploaded_file in uploaded_files or []:
                temp_path = _write_upload_to_temp(uploaded_file, temp_dir)
                temp_paths.append(temp_path)
                file_paths.append(str(temp_path))

            state_input: Dict[str, Any] = {
                "messages": [HumanMessage(content=query)],
                "file_paths": file_paths,
            }

            result = graph.invoke(
                state_input,
                config={
                    "configurable": {"thread_id": session_id},
                    "run_name": "dsai_agent_graph",
                    "tags": ["dsai-agent", "request"],
                    "metadata": {
                        "session_id": session_id,
                        "uploaded_file_count": len(uploaded_files or []),
                    },
                },
            )

            extracted_text_map = result.get("extracted_text_map", {}) or {}
            retrieved_context = result.get("retrieved_context")
            final_message = result["messages"][-1].content
            final_output = final_message if isinstance(final_message, str) else str(final_message)

            return {
                "status": "success",
                "session_id": session_id,
                "query": query,
                "files_received": [getattr(f, "filename", "") for f in (uploaded_files or [])],
                "extracted_text_map": extracted_text_map,
                "retrieved_context": retrieved_context,
                "plan_trace": result.get("plan_trace", []),
                "final_output": final_output,
                "extracted_text": _format_extracted_text(extracted_text_map),
            }
    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def run_agent_request(
    query: str,
    session_id: str,
    uploaded_files: Optional[List[object]] = None,
) -> Dict[str, object]:
    return _execute_graph_request(query, session_id, uploaded_files)


def stream_agent_request(
    query: str,
    session_id: str,
    uploaded_files: Optional[List[object]] = None,
) -> Iterator[str]:
    event_queue: Queue = Queue()
    result_holder: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            result_holder["payload"] = _execute_graph_request(
                query=query,
                session_id=session_id,
                uploaded_files=uploaded_files,
            )
            event_queue.put({"type": "result", "payload": result_holder["payload"]})
        except Exception as exc:
            event_queue.put({"type": "error", "text": str(exc)})
        finally:
            event_queue.put({"type": "done"})

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    yield json.dumps({"type": "status", "text": "Agent is thinking..."}) + "\n"

    while True:
        try:
            event = event_queue.get(timeout=0.25)
        except Empty:
            if not worker.is_alive():
                break
            continue

        event_type = event.get("type")
        if event_type == "token":
            yield json.dumps(event) + "\n"
        elif event_type in {"status", "error", "result"}:
            yield json.dumps(event) + "\n"
        elif event_type == "done":
            break
