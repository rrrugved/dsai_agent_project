from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State schema for the multi-input agent workflow with Self-Reflection.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    file_paths: List[str]
    intent_type: Optional[str]
    needs_clarification: bool
    extracted_text_map: Dict[str, str]
    retrieved_context: Optional[str]
    plan_trace: List[str]
    task_category: Optional[str]
    is_context_relevant: Optional[bool]
    stream_queue: Optional[Any]
