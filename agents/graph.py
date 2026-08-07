import os
import re
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.tools import (
    extract_text_from_image,
    transcribe_audio,
    fetch_webpage_content,
    fetch_youtube_transcript,
    parse_pdf
)

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def _detect_urls(text: str) -> List[str]:
    """Extracts all HTTP/HTTPS URLs from a given text string."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return re.findall(url_pattern, text)

def _is_youtube_url(url: str) -> bool:
    """Validates if a parsed URL belongs to the YouTube domain."""
    return "youtube.com" in url.lower() or "youtu.be" in url.lower()

def ingest_and_classify(state: AgentState) -> Dict[str, Any]:
    """Classifies user intent based on query text and file attachments to determine the routing path."""
    messages = state.get("messages", [])
    file_paths = state.get("file_paths", [])
    
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            last_user_message = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    has_files = len(file_paths) > 0
    has_urls = len(_detect_urls(last_user_message)) > 0
    clean_prompt = last_user_message.strip()

    if (has_files or has_urls) and len(clean_prompt) < 3:
        return {
            "intent_type": "ambiguous",
            "needs_clarification": True,
            "plan_trace": ["Input received: Files/URLs detected without specific instructions."],
            "task_category": None
        }

    if not has_files and not has_urls:
        return {
            "intent_type": "conversational",
            "needs_clarification": False,
            "plan_trace": ["Input received: Standard text query. Routing to Direct LLM."],
            "task_category": "qa"
        }

    prompt_lower = clean_prompt.lower()
    task = "qa"
    if "summar" in prompt_lower:
        task = "summarization"
    elif "sentiment" in prompt_lower:
        task = "sentiment"
    elif "code" in prompt_lower or "bug" in prompt_lower or "explain" in prompt_lower:
        task = "code_explanation"

    return {
        "intent_type": "tool_required",
        "needs_clarification": False,
        "plan_trace": [f"Input received: Media/URL detected. Task identified: {task}"],
        "task_category": task
    }

def followup_node(state: AgentState) -> Dict[str, Any]:
    """Generates a clarifying question when the user prompt is ambiguous or extracted context is irrelevant."""
    file_paths = state.get("file_paths", [])
    is_relevant = state.get("is_context_relevant")
    
    if is_relevant is False:
        question = (
            "I processed your provided file(s), but the extracted content does not seem to contain "
            "relevant information to answer your request. Could you please clarify or provide additional details?"
        )
    elif file_paths:
        question = (
            f"I see you uploaded {len(file_paths)} file(s). Could you please clarify what you would like me to do with them? "
            "(e.g., summarize, extract action items, perform sentiment analysis, or explain code)"
        )
    else:
        question = "Could you please provide more details or clarify what task you would like me to perform?"

    return {
        "messages": [AIMessage(content=question)],
        "plan_trace": state.get("plan_trace", []) + ["Action: Prompted user for clarification."]
    }

def direct_llm_node(state: AgentState) -> Dict[str, Any]:
    """Bypasses tool execution to directly answer conversational queries."""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "plan_trace": state.get("plan_trace", []) + ["Action: Answered query directly using internal LLM knowledge."]
    }

def planner_and_tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes the appropriate data extraction tools based on input file types and detected URLs."""
    file_paths = state.get("file_paths", [])
    messages = state.get("messages", [])
    
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            last_user_message = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    extracted_map = dict(state.get("extracted_text_map", {}))
    trace = list(state.get("plan_trace", []))

    for path in file_paths:
        filename = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            trace.append(f"Executing Tool: parse_pdf on '{filename}'")
            res = parse_pdf.invoke({"file_path": path})
            extracted_map[filename] = res

        elif ext in [".png", ".jpg", ".jpeg"]:
            trace.append(f"Executing Tool: extract_text_from_image (OCR) on '{filename}'")
            res = extract_text_from_image.invoke({"file_path": path})
            extracted_map[filename] = res

        elif ext in [".mp3", ".wav", ".m4a"]:
            trace.append(f"Executing Tool: transcribe_audio (Whisper) on '{filename}'")
            res = transcribe_audio.invoke({"file_path": path})
            extracted_map[filename] = res

    combined_corpus = last_user_message + " " + " ".join(extracted_map.values())
    found_urls = _detect_urls(combined_corpus)

    for url in found_urls:
        if _is_youtube_url(url):
            trace.append(f"Executing Tool: fetch_youtube_transcript for '{url}'")
            res = fetch_youtube_transcript.invoke({"url_or_id": url})
            extracted_map[f"YouTube: {url}"] = res
        else:
            trace.append(f"Executing Tool: fetch_webpage_content for '{url}'")
            res = fetch_webpage_content.invoke({"url": url})
            extracted_map[f"Web: {url}"] = res

    return {
        "extracted_text_map": extracted_map,
        "plan_trace": trace
    }

def relevancy_checker_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates whether the extracted document context contains relevant information to answer the user's query."""
    extracted_map = state.get("extracted_text_map", {})
    messages = state.get("messages", [])
    
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            last_user_message = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    if not extracted_map:
        return {"is_context_relevant": False}

    combined_text = "\n\n".join([f"--- {src} ---\n{content}" for src, content in extracted_map.items()])

    checker_prompt = f"""
    You are a strict relevance grader. Evaluate whether the extracted context below contains sufficient or relevant information to answer or address the user query.

    USER QUERY: {last_user_message}

    EXTRACTED CONTEXT:
    {combined_text[:3000]}

    Respond with ONLY 'YES' if the context is relevant and useful, or 'NO' if it is irrelevant, corrupt, or empty.
    """

    res = llm.invoke([HumanMessage(content=checker_prompt)])
    answer = res.content.strip().upper()
    is_relevant = "YES" in answer

    trace = state.get("plan_trace", [])
    trace.append(f"Self-Reflection Check: Context relevance evaluated as {is_relevant}.")

    return {
        "is_context_relevant": is_relevant,
        "plan_trace": trace
    }

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Formats the final LLM response according to strict task-specific guidelines (e.g., summarization, code explanation)."""
    task = state.get("task_category", "qa")
    extracted_map = state.get("extracted_text_map", {})
    messages = state.get("messages", [])
    
    last_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            last_user_message = msg.content if hasattr(msg, "content") else msg.get("content", "")
            break

    context_str = "\n\n".join([f"Source ({src}):\n{content}" for src, content in extracted_map.items()])

    formatting_instructions = ""
    if task == "summarization":
        formatting_instructions = """
        You MUST structure your output strictly as follows:
        1. **1-line Summary:** <One concise sentence summarizing the core content>
        2. **Key Highlights:**
           - <Bullet 1>
           - <Bullet 2>
           - <Bullet 3>
        3. **Detailed Summary:** <A comprehensive exactly 5-sentence summary paragraph>
        """
    elif task == "sentiment":
        formatting_instructions = """
        You MUST structure your output strictly as follows:
        - **Sentiment Label:** <Positive / Negative / Neutral / Mixed>
        - **Confidence Score:** <0% to 100%>
        - **Justification:** <One clear sentence explaining the reasoning>
        """
    elif task == "code_explanation":
        formatting_instructions = """
        You MUST structure your output strictly as follows:
        - **Code Functionality:** <Detailed breakdown of what the code does>
        - **Bug Analysis & Warnings:** <Identified bugs, syntax issues, or logical flaws (or state 'None detected')>
        - **Time & Space Complexity:** <Estimated Big-O complexity analysis>
        """

    system_instructions = f"""
    You are an expert multi-modal AI agent. Answer the user request using the provided context.
    All outputs must be text-only.

    {formatting_instructions}

    Context:
    {context_str}
    """

    full_messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=last_user_message)
    ]

    response = llm.invoke(full_messages)

    return {
        "messages": [response],
        "plan_trace": state.get("plan_trace", []) + ["Action: Synthesized final structured answer."]
    }

def route_after_classify(state: AgentState) -> Literal["followup_node", "direct_llm_node", "planner_and_tool_node"]:
    """Determines the next graph node based on the classified intent type."""
    intent = state.get("intent_type")
    if intent == "ambiguous":
        return "followup_node"
    elif intent == "conversational":
        return "direct_llm_node"
    return "planner_and_tool_node"

def route_after_relevancy(state: AgentState) -> Literal["synthesizer_node", "followup_node"]:
    """Routes execution to synthesis or follow-up based on context relevancy."""
    if state.get("is_context_relevant") is False:
        return "followup_node"
    return "synthesizer_node"


builder = StateGraph(AgentState)

builder.add_node("ingest_and_classify", ingest_and_classify)
builder.add_node("followup_node", followup_node)
builder.add_node("direct_llm_node", direct_llm_node)
builder.add_node("planner_and_tool_node", planner_and_tool_node)
builder.add_node("relevancy_checker_node", relevancy_checker_node)
builder.add_node("synthesizer_node", synthesizer_node)

builder.add_edge(START, "ingest_and_classify")

builder.add_conditional_edges(
    "ingest_and_classify",
    route_after_classify,
    {
        "followup_node": "followup_node",
        "direct_llm_node": "direct_llm_node",
        "planner_and_tool_node": "planner_and_tool_node"
    }
)

builder.add_edge("planner_and_tool_node", "relevancy_checker_node")

builder.add_conditional_edges(
    "relevancy_checker_node",
    route_after_relevancy,
    {
        "synthesizer_node": "synthesizer_node",
        "followup_node": "followup_node"
    }
)

builder.add_edge("followup_node", END)
builder.add_edge("direct_llm_node", END)
builder.add_edge("synthesizer_node", END)

graph = builder.compile()