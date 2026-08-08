from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Response contract shared by the FastAPI backend and Streamlit frontend."""

    status: str
    session_id: str
    query: Optional[str] = None
    files_received: List[str] = Field(default_factory=list)
    extracted_text_map: Dict[str, str] = Field(default_factory=dict)
    retrieved_context: Optional[str] = None
    plan_trace: List[str] = Field(default_factory=list)
    final_output: str
